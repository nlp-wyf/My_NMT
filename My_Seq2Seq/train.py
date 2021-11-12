import time
import random
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from My_Seq2Seq.Seq2Seq.model import Encoder, Decoder, Seq2Seq
from My_Seq2Seq.Seq2Seq.model_with_attention import AttnDecoder, Seq2SeqAtt
from My_Seq2Seq.data_loader import TranslationDataset
from My_Seq2Seq.config import load_config

config = load_config()


# calculate the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_network(model, method='xavier'):
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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, data_loader, optimizer, teacher_forcing_ratio=0.5):
    model.predict = False
    model.train()

    epoch_loss = 0
    for batch in tqdm(data_loader):
        # shape = [seq_len, batch]
        input_batchs = batch["src"]
        target_batchs = batch["trg"]
        # list
        input_lens = batch["src_len"]
        target_lens = batch["trg_len"]

        loss = model(input_batchs, input_lens, target_batchs, target_lens, teacher_forcing_ratio)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)

    return epoch_loss / len(data_loader)


def evaluate(model, data_loader):
    model.predict = False
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # shape = [seq_len, batch]
            input_batchs = batch["src"]
            target_batchs = batch["trg"]
            # list
            input_lens = batch["src_len"]
            target_lens = batch["trg_len"]

            loss = model(input_batchs, input_lens, target_batchs, target_lens, teacher_forcing_ratio=0)

            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def run():
    # 数据集
    train_set = TranslationDataset(config.data_path)
    logger.info("-------- Dataset Build! --------")
    train_loader = DataLoader(train_set, batch_size=config.batch_size, collate_fn=train_set.padding_batch)
    logger.info("-------- Get Dataloader! --------")

    # 词表
    src_vocab_size = len(train_set.en2id)
    tgt_vocab_size = len(train_set.ch2id)
    logger.info("-------- Load Vocabulary! --------")

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
    logger.info("-------- Initialize Model! --------")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练
    logger.info("-------- Start Train! --------")
    best_valid_loss = float('inf')
    for epoch in range(config.num_epochs):

        start_time = time.time()
        train_loss = train(model, train_loader, optimizer)
        valid_loss = evaluate(model, train_loader)
        end_time = time.time()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), config.model_path)

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        logger.info(
            f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}')

    logger.debug(f"best valid loss: {best_valid_loss:.3f}")
    logger.info("-------- Train Done! --------")


if __name__ == '__main__':
    # 设定随机种子，保持前后结果的一致性
    SEED = 2021
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    run()
