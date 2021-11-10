import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader

from My_Seq2Seq.Seq2Seq.model import Seq2Seq, Decoder, Encoder
from My_Seq2Seq.config import load_config
from My_Seq2Seq.data_loader import TranslateDataset

config = load_config()


# calculate the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# init weights
def init_network(model, method='xavier'):
    # 权重初始化，默认xavier
    for name, param in model.named_parameters():
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_normal_(param)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(param)
            else:
                nn.init.normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
        else:
            pass


# 训练
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    # for i, batch in enumerate(iterator):
    for batch in tqdm(iterator):
        enc_input, dec_input, target = batch
        # src.shape [seq_len, batch_size]

        enc_input = enc_input.transpose(1, 0).to(config.device)
        dec_input = dec_input.transpose(1, 0).to(config.device)
        target = target.transpose(1, 0).to(config.device)

        optimizer.zero_grad()

        output = model.forward(enc_input, dec_input)
        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output.view(-1, output.shape[-1])
        target = target.view(-1)

        # output = [(trg sent len - 1) * batch size, output dim]
        # trg = [(trg sent len - 1) * batch size]

        loss = criterion(output, target)

        loss.backward()

        # gradient clipping 防止梯度爆炸问题
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 测试
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        # for i, batch in enumerate(iterator):
        for batch in tqdm(iterator):
            enc_input, dec_input, target = batch

            enc_input = enc_input.transpose(1, 0).to(config.device)
            dec_input = dec_input.transpose(1, 0).to(config.device)
            target = target.transpose(1, 0).to(config.device)

            output = model.forward(enc_input, dec_input, 0)

            output = output.view(-1, output.shape[-1])
            target = target.view(-1)

            loss = criterion(output, target)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def test(model, test_iterator, criterion):
    model.load_state_dict(torch.load(config.model_path))
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f}')


def run(model, train_iterator, dev_iterator, test_iterator):
    best_dev_loss = float('inf')

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # criterion
    # we ignore the loss whenever the target token is a padding token
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD)

    for epoch in range(config.num_epochs):

        train_loss = train(model, train_iterator, optimizer, criterion, config.CLIP)
        dev_loss = evaluate(model, dev_iterator, criterion)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), config.model_path)

        logger.info("Epoch: {}, Train loss: {}, Dev loss: {}".format(epoch + 1, train_loss, dev_loss))

    test(model, test_iterator, criterion)


if __name__ == '__main__':
    # set the random seeds for deterministic 14 results
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # 数据预处理
    train_dataset = TranslateDataset(config.train_file)
    dev_dataset = TranslateDataset(config.dev_file)
    test_dataset = TranslateDataset(config.test_file)
    logger.info("-------- Dataset Build! --------")

    train_iterator = DataLoader(train_dataset, batch_size=config.batch_size,
                                shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_iterator = DataLoader(dev_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=dev_dataset.collate_fn)
    test_iterator = DataLoader(dev_dataset, batch_size=config.batch_size,
                               shuffle=False, collate_fn=test_dataset.collate_fn)
    logger.info("-------- Get Dataloader! --------")

    enc = Encoder(config.src_vocab_size, config.embed_dim, config.hidden_dim, config.num_layers, config.dropout)
    dec = Decoder(config.tgt_vocab_size, config.embed_dim, config.hidden_dim, config.num_layers, config.dropout)
    model = Seq2Seq(enc, dec, config.device)
    init_network(model)
    model = model.to(config.device)
    logger.info("-------- Initialize Model! --------")

    logger.info("-------- Start Train! --------")
    run(model, train_iterator, dev_iterator, test_iterator)
    logger.info("-------- Train Done! --------")
