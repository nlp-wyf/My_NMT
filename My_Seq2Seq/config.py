import torch
import attrdict


def load_config():
    CONF = {
        'PAD': 0,
        'UNK': 1,
        'BOS': 2,
        'EOS': 3,
        'batch_size': 64,  # 每批次训练数据数量  32  64
        'src_vocab_size': 4000,  # 32000
        'tgt_vocab_size': 4000,  # 32000
        'num_epochs': 20,  # 训练轮数  40
        'require_improvement': 1000,  # 若超过1000个batch效果还没有提升，则提前结束训练

        # greedy decode的最大句子长度
        'max_len': 60,  # 最大句子长度

        'CLIP': 1,
        # transformer model structure parameter
        'learning_rate': 0.001,  # 0.001  3e-4
        'dropout': 0.5,  # dropout比例
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,

        'train_file': 'parallel_corpus/en_zh/train.txt',  # 训练集数据文件
        'dev_file': "parallel_corpus/en_zh/dev.txt",  # 验证(开发)集数据文件
        'test_file': "parallel_corpus/en_zh/test.txt",

        'model_path': 'saved_dict/model.ckpt',  # 模型保存路径(注意如当前目录无save文件夹需要自己创建)
        'log_path': 'log/transformer',
        'output_path': 'translate.txt',

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # 设备

    }

    CONF = attrdict.AttrDict(CONF)
    return CONF
