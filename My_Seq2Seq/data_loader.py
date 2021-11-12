import torch
from torch.utils.data import Dataset, DataLoader

from My_Seq2Seq.config import load_config

config = load_config()


class TranslationDataset(Dataset):
    def __init__(self, data_path):
        self.en_num_data, self.ch_num_data, self.en2id, \
        self.id2en, self.ch2id, self.id2ch, self.basic_dict = self.build_data(data_path)

        assert len(self.en_num_data) == len(self.ch_num_data)

    def __len__(self):
        return len(self.en_num_data)

    def __getitem__(self, idx):
        src_sample = self.en_num_data[idx]
        src_len = len(self.en_num_data[idx])
        trg_sample = self.ch_num_data[idx]
        trg_len = len(self.ch_num_data[idx])
        return {"src": src_sample, "src_len": src_len, "trg": trg_sample, "trg_len": trg_len}

    def build_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()
        data = data.strip()
        data = data.split('\n')

        # 分割英文数据和中文数据
        en_data = [line.split('\t')[0] for line in data]
        ch_data = [line.split('\t')[1] for line in data]

        # 按字符级切割，并添加<eos>
        en_token_list = [[char for char in line] + ["<eos>"] for line in en_data]
        ch_token_list = [[char for char in line] + ["<eos>"] for line in ch_data]

        # 基本字典
        basic_dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        # 分别生成中英文字典
        en_vocab = set(''.join(en_data))
        en2id = {char: i + len(basic_dict) for i, char in enumerate(en_vocab)}
        en2id.update(basic_dict)
        id2en = {v: k for k, v in en2id.items()}

        # 分别生成中英文字典
        ch_vocab = set(''.join(ch_data))
        ch2id = {char: i + len(basic_dict) for i, char in enumerate(ch_vocab)}
        ch2id.update(basic_dict)
        id2ch = {v: k for k, v in ch2id.items()}

        # 利用字典，映射数据
        en_num_data = [[en2id[en] for en in line] for line in en_token_list]
        ch_num_data = [[ch2id[ch] for ch in line] for line in ch_token_list]

        return en_num_data, ch_num_data, en2id, id2en, ch2id, id2ch, basic_dict

    def padding_batch(self, batch):
        """
        input: -> list of dict
            [{'src': [1, 2, 3], 'trg': [1, 2, 3]}, {'src': [1, 2, 2, 3], 'trg': [1, 2, 2, 3]}]
        output: -> dict of tensor
            {
                "src": [[1, 2, 3, 0], [1, 2, 2, 3]].T
                "trg": [[1, 2, 3, 0], [1, 2, 2, 3]].T
            }
        """
        src_lens = [d["src_len"] for d in batch]
        trg_lens = [d["trg_len"] for d in batch]

        src_max = max([d["src_len"] for d in batch])
        trg_max = max([d["trg_len"] for d in batch])
        for d in batch:
            d["src"].extend([self.en2id["<pad>"]] * (src_max - d["src_len"]))
            d["trg"].extend([self.ch2id["<pad>"]] * (trg_max - d["trg_len"]))

        srcs = torch.tensor([pair["src"] for pair in batch], dtype=torch.long, device=config.device)
        trgs = torch.tensor([pair["trg"] for pair in batch], dtype=torch.long, device=config.device)

        batch = {"src": srcs.T, "src_len": src_lens, "trg": trgs.T, "trg_len": trg_lens}
        # batch.src/trg的shape为[seq_len, batch_size]
        return batch


if __name__ == '__main__':

    # 数据集
    train_set = TranslationDataset(config.data_path)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, collate_fn=train_set.padding_batch)
    for i, batch in enumerate(train_loader):
        if i == 10:
            srcs = batch['src']
            tgts = batch['trg']
            print(srcs.shape)
