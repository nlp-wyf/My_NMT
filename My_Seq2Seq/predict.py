import random
import torch

from My_Seq2Seq.Seq2Seq.model import Encoder, Decoder, Seq2Seq
from My_Seq2Seq.Seq2Seq.model_with_attention import AttnDecoder, Seq2SeqAtt
from My_Seq2Seq.config import load_config
from My_Seq2Seq.data_loader import TranslationDataset

config = load_config()


def translate(model, sample, idx2token=None):
    model.predict = True
    model.eval()

    # shape = [seq_len, 1]
    input_batch = sample["src"]
    # list
    input_len = sample["src_len"]

    output_tokens = model(input_batch, input_len)
    output_tokens = [idx2token[t] for t in output_tokens]

    return "".join(output_tokens)


def predict():
    train_set = TranslationDataset(config.data_path)

    # 词表
    src_vocab_size = len(train_set.en2id)
    tgt_vocab_size = len(train_set.ch2id)

    # 初始化模型
    attn_method = "general"
    encoder = Encoder(src_vocab_size, config.enc_embedding_dim,
                      config.hidden_dim, config.num_layers, config.enc_dropout)
    # decoder = Decoder(tgt_vocab_size, config.dec_embedding_dim,
    #                   config.hidden_dim, config.num_layers, config.dec_dropout)
    decoder = AttnDecoder(tgt_vocab_size, config.dec_embedding_dim, config.hidden_dim,
                          config.num_layers, config.dec_dropout, attn_method)
    # model = Seq2Seq(encoder, decoder, config.device, basic_dict=train_set.basic_dict)
    model = Seq2SeqAtt(encoder, decoder, config.device, basic_dict=train_set.basic_dict)

    model = model.to(config.device)

    model.load_state_dict(torch.load(config.model_path))

    en_num_data = train_set.en_num_data
    ch_num_data = train_set.ch_num_data
    id2en = train_set.id2en
    id2ch = train_set.id2ch
    seed = 2021
    random.seed(seed)
    for i in random.sample(range(len(en_num_data)), 10):  # 随机看10个
        en_tokens = list(filter(lambda x: x != 0, en_num_data[i]))  # 过滤零
        ch_tokens = list(filter(lambda x: x != 3 and x != 0, ch_num_data[i]))  # 和机器翻译作对照
        sentence = [id2en[t] for t in en_tokens]
        print("【原文】")
        print("".join(sentence))
        translation = [id2ch[t] for t in ch_tokens]
        print("【原文】")
        print("".join(translation))
        test_sample = {}
        test_sample["src"] = torch.tensor(en_tokens, dtype=torch.long, device=config.device).reshape(-1, 1)
        test_sample["src_len"] = [len(en_tokens)]
        print("【机器翻译】")
        print(translate(model, test_sample, id2ch), end="\n\n")


if __name__ == '__main__':
    predict()
