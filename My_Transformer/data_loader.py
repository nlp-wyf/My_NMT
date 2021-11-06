import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from My_Transformer.utils import english_tokenizer_load, chinese_tokenizer_load
from My_Transformer.config import load_config

config = load_config()
DEVICE = config.device


def padding_mask(seq, pad_idx):
    # 对于当前输入的句子非空部分进行判断成bool序列
    # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
    # seq.shape [batch_size, seq_len]
    # 返回的是形状为[batch_size, 1, seq_len]的bool矩阵
    return (seq != pad_idx).unsqueeze(-2)


def sequence_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    batch_size, seq_len = seq.size()
    attn_shape = (batch_size, seq_len, seq_len)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    # 此时右上角(不含主对角线)为全0，左下角(含主对角线)为全1的subsequent_mask矩阵
    mask = 1 - mask
    # 此时右上角(不含主对角线)为False，左下角(含主对角线)为True的subsequent_mask矩阵
    mask = mask.bool()
    # [batch_size, tgt_len, tgt_len]
    return mask


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src_text, tgt_text, src, tgt=None, pad=0):
        self.src_text = src_text
        self.tgt_text = tgt_text
        src = src.to(DEVICE)
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = padding_mask(src, pad_idx=pad)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if tgt is not None:
            tgt = tgt.to(DEVICE)
            # decoder要用到的target输入部分
            self.tgt = tgt[:, :-1]
            # decoder训练时应预测输出的target结果
            self.tgt_y = tgt[:, 1:]
            # 将target输入部分进行attention mask
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.n_tokens = (self.tgt_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = padding_mask(tgt, pad_idx=pad)
        tgt_mask = tgt_mask & Variable(sequence_mask(tgt).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        self.out_en_sent, self.out_zh_sent = self.get_dataset(data_path, sort=True)
        self.sp_en = english_tokenizer_load()
        self.sp_zh = chinese_tokenizer_load()
        self.PAD = self.sp_en.pad_id()  # 0
        self.BOS = self.sp_en.bos_id()  # 2
        self.EOS = self.sp_en.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """把中文和英文按照同样的顺序排序, 以英文句子长度排序的(句子下标)顺序为基准"""
        # dataset = json.load(open(data_path, 'r', encoding='utf-8'))
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                en, zh = line.strip().split('\t')
                dataset.append([en, zh])
        out_en_sent = []
        out_zh_sent = []
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])
            out_zh_sent.append(dataset[idx][1])
        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_zh_sent = [out_zh_sent[i] for i in sorted_index]
        return out_en_sent, out_zh_sent

    def __getitem__(self, idx):
        en_text = self.out_en_sent[idx]
        zh_text = self.out_zh_sent[idx]
        return [en_text, zh_text]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_en.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_zh.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # pad_sequence对一个batch批次(以单词id表示)的数据进行padding填充对齐长度
        # 返回[batch, N], N为当前batch最大长度
        batch_source = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                    batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_source, batch_target, self.PAD)
