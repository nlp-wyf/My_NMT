import random
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0  # 开始标志
EOS_token = 1  # 结束标志
MAX_LENGTH = 20  # 最大长度


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout=0.5):
        """

        :param input_dim:  输入源词库的大小
        :param embed_dim:   输入单词Embedding的维度
        :param hidden_dim:   隐层的维度
        :param num_layers:   隐层的层数
        :param dropout:  dropout参数
        """
        super(Encoder, self).__init__()
        # 设置输入参数
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 根据input和embbed的的维度，初始化embedding层
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # 初始化GRU，获取embbed的输入维度，输出隐层的维度，设置GRU层的参数
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        # src.shape [src_seq_len, batch_size] 这句话的长度和batch大小

        # embedded.shape [src_seq_len, batch_size, embed_dim]
        # embedded = self.embedding(src).view(1, 1, -1)
        embedded = self.dropout(self.embedding(src))

        # outputs.shape [src_seq_len, batch_size, hidden_dim * num_directions]
        # hidden.shape [num_layers * num_directions, batch_size, hidden_dim]
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout=0.5):
        """

        :param output_dim: 输入目标词库的大小
        :param embed_dim: 输入单词Embedding的维度
        :param hidden_dim: 隐层的维度
        :param num_layers: 隐层的层数
        :param dropout: dropout参数
        """
        super(Decoder, self).__init__()

        # 设置参数
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        # 以适当的维度，初始化每一层。
        # decoder层由embedding, GRU, 线性层和Log softmax 激活函数组成
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_dim, output_dim)  # 线性层

    def forward(self, tgt, hidden):
        # tgt.shape为[1, batch_size]
        # reshape the tgt to (1, batch_size)
        tgt = tgt.unsqueeze(0)
        # tgt = tgt.view(1, -1)

        # embedded.shape [1, batch_size, embed_dim]
        # embedded = F.relu(self.embedding(tgt))
        embedded = self.dropout(self.embedding(tgt))

        # output.shape [1, batch_size, hidden_dim * num_directions]
        # hidden.shape [num_layers * num_direction, batch_size, hidden_dim]
        output, hidden = self.gru(embedded, hidden)

        # output[0].shape == output.squeeze(0)== [batch_size, output_dim]
        # prediction = self.softmax(self.out(output[0]))
        prediction = self.out(output.squeeze(0))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        # 初始化encoder和decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source.shape [src_seq_len, batch_size]
        # target.shape [tgt_seq_len, batch_size]

        batch_size = target.shape[1]
        target_length = target.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(target_length, batch_size, tgt_vocab_size).to(self.device)

        # 为语句中的每一个word编码
        # for i in range(source_length):
        #     encoder_output, encoder_hidden = self.encoder(source[i])

        encoder_output, encoder_hidden = self.encoder(source)

        # 使用encoder的hidden层作为decoder的hidden层
        decoder_hidden = encoder_hidden.to(device)

        # 在预测前，添加一个token
        # decoder_input = torch.tensor([SOS_token], device=device)
        # decoder的第一个输入是开始标志<sos>
        decoder_input = target[0, :]

        for t in range(1, target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            # 是否使用Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio

            top1 = decoder_output.argmax(1)

            decoder_input = target[t] if teacher_force else top1

        return outputs
