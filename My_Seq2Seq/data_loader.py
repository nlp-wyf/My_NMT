import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from My_Seq2Seq.config import load_config
from My_Seq2Seq.utils import english_tokenizer_load, chinese_tokenizer_load

config = load_config()
DEVICE = config.device


class TranslateDataset(Dataset):
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

        # decoder要用到的target输入部分
        dec_input = batch_target[:, :-1]
        # decoder训练时应预测输出的target结果
        target = batch_target[:, 1:]

        return batch_source, dec_input, target


if __name__ == '__main__':
    # 数据预处理
    dev_dataset = TranslateDataset(config.dev_file)
    loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, collate_fn=dev_dataset.collate_fn)
    for i, data in enumerate(loader):
        if i == 14:
            enc_input, dec_input, target = data
            print(enc_input.shape)
            print(dec_input.shape)
            print(target.shape)
