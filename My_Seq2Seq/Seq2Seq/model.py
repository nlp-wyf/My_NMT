import random
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout)

    def forward(self, input_seqs, input_lengths, hidden):
        # input_seqs = [seq_len, batch]
        embedded = self.embedding(input_seqs)
        # embedded = [seq_len, batch, embed_dim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)

        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs = [seq_len, batch, hid_dim * n directions]
        # output_lengths = [batch]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(Decoder, self).__init__()

        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout)

        self.fc_out = nn.Linear(hidden_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, token_inputs, hidden):
        # token_inputs = [batch]
        token_inputs = token_inputs.unsqueeze(0)
        # [1, batch_size]
        batch_size = token_inputs.size(1)
        embedded = self.dropout(self.embedding(token_inputs).view(1, batch_size, -1))
        # embedded = [1, batch, emb_dim]

        output, hidden = self.gru(embedded, hidden)
        # output = [1, batch,  n_directions * hid_dim]
        # hidden = [n_layers * n_directions, batch, hid_dim]

        output = self.fc_out(output.squeeze(0))
        output = self.softmax(output)
        # output = [batch, tgt_vocab_size]
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, predict=False, basic_dict=None, max_len=100):
        super(Seq2Seq, self).__init__()

        self.device = device

        self.encoder = encoder
        self.decoder = decoder

        self.predict = predict  # 训练阶段还是预测阶段
        self.basic_dict = basic_dict  # decoder的字典，存放特殊token对应的id
        self.max_len = max_len  # 翻译时最大输出长度

        assert encoder.hidden_dim == decoder.hidden_dim
        assert encoder.num_layers == decoder.num_layers

    def forward(self, input_batches, input_lengths, target_batches=None, target_lengths=None,
                teacher_forcing_ratio=0.5):
        # input_batches = target_batches = [seq_len, batch]
        batch_size = input_batches.size(1)

        BOS_token = self.basic_dict["<bos>"]
        EOS_token = self.basic_dict["<eos>"]
        PAD_token = self.basic_dict["<pad>"]

        # 初始化
        encoder_hidden = torch.zeros(self.encoder.num_layers, batch_size,
                                     self.encoder.hidden_dim, device=self.device)

        # encoder_output = [seq_len, batch, hid_dim * n directions]
        # encoder_hidden = [n_layers*n_directions, batch, hid_dim]
        encoder_output, encoder_hidden = self.encoder(input_batches, input_lengths, encoder_hidden)

        # 初始化
        decoder_input = torch.tensor([BOS_token] * batch_size, dtype=torch.long, device=self.device)
        # 将encoder的状态值作为decoder的初始状态
        decoder_hidden = encoder_hidden

        if self.predict:
            # 预测阶段使用
            # 一次只输入一句话
            assert batch_size == 1
            output_tokens = []

            while True:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # topi.shape [1, 1]
                topv, topi = decoder_output.topk(1)
                # 上一个预测作为下一个输入
                decoder_input = topi.squeeze(1)
                output_token = topi.squeeze().detach().item()
                if output_token == EOS_token or len(output_tokens) == self.max_len:
                    break
                output_tokens.append(output_token)
            return output_tokens

        else:
            # 训练阶段
            max_target_length = max(target_lengths)
            all_decoder_outputs = torch.zeros((max_target_length, batch_size, self.decoder.tgt_vocab_size),
                                              device=self.device)

            for t in range(max_target_length):
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    # decoder_output = [batch, tgt_vocab_size]
                    # decoder_hidden = [n_layers * n_directions, batch, hid_dim]
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    all_decoder_outputs[t] = decoder_output
                    decoder_input = target_batches[t]  # 下一个输入来自训练数据
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    # topi.shape [batch, 1]
                    topv, topi = decoder_output.topk(1)
                    all_decoder_outputs[t] = decoder_output
                    # 下一个输入来自模型预测
                    decoder_input = topi.squeeze(1)

            loss_fn = nn.NLLLoss(ignore_index=PAD_token)
            loss = loss_fn(
                # [batch_size * seq_len, tgt_vocab_size]
                all_decoder_outputs.reshape(-1, self.decoder.tgt_vocab_size),
                # [batch_size * seq_len]
                target_batches.reshape(-1)
            )
            return loss
