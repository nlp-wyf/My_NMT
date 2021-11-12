import torch
import attrdict


def load_config():
    CONF = {
        'PAD': 0,
        'UNK': 1,
        'BOS': 2,
        'EOS': 3,
        'batch_size': 64,  # 每批次训练数据数量  32  64
        'num_epochs': 2,  # 训练轮数  40
        'CLIP': 1,
        'learning_rate': 1e-4,  # 0.001  1e-4
        'enc_dropout': 0.5,  # dropout比例
        'dec_dropout': 0.5,
        'enc_embedding_dim': 256,
        'dec_embedding_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,

        'train_file': 'parallel_corpus/en_zh/train.txt',  # 训练集数据文件
        'dev_file': "parallel_corpus/en_zh/dev.txt",  # 验证(开发)集数据文件
        'test_file': "parallel_corpus/en_zh/test.txt",

        'data_path': 'parallel_corpus/en_zh/en_zh.txt',

        'model_path': 'saved_dict/en2zh_model.pt',  # 模型保存路径(注意如当前目录无save文件夹需要自己创建)
        'log_path': 'log/transformer',
        'output_path': 'translate.txt',

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # 设备

    }

    CONF = attrdict.AttrDict(CONF)
    return CONF
