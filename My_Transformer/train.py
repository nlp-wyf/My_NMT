import sacrebleu
from tqdm import tqdm
from loguru import logger

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from My_Transformer.Transformer.beam_search import beam_search
from My_Transformer.Transformer.greedy_search import batch_greedy_decode
from My_Transformer.Transformer.loss_func import LossCompute, MultiGPULossCompute
from My_Transformer.Transformer.warm_up_optimizer import get_std_opt
from My_Transformer.Transformer.normalization import LabelSmoothing
from My_Transformer.config import load_config
from My_Transformer.Transformer.models import make_model
from My_Transformer.data_loader import MTDataset, padding_mask
from My_Transformer.utils import chinese_tokenizer_load

config = load_config()
# writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))


def run_epoch(model, data, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.n_tokens)

        total_loss += loss
        total_tokens += batch.n_tokens

    return total_loss / total_tokens


def train(model, train_data, dev_data, criterion, optimizer):
    """
    训练并保存模型
    """
    best_bleu_score = 0.0
    # 初始化模型在dev集上的最优Loss为一个较大值
    # best_dev_loss = 1e5

    for epoch in range(1, config.num_epochs + 1):
        # 模型训练
        model.train()
        train_loss = run_epoch(model, train_data, LossCompute(model.generator, criterion, optimizer))
        # 多gpu训练(如果有的话)
        # train_loss = run_epoch(model_par, train_data,
        #                        MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))
        logger.info("Epoch: {}, Train loss: {}".format(epoch, train_loss))
        model.eval()
        # 在dev集上进行loss评估
        dev_loss = run_epoch(model, dev_data, LossCompute(model.generator, criterion, None))
        # dev_loss = run_epoch(model_par, dev_data,
        #                      MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(model, dev_data, use_beam=False)
        logger.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        # writer.add_scalar("Train loss", train_loss, global_step=epoch)
        # writer.add_scalar("Dev loss", dev_loss, global_step=epoch)

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if bleu_score > best_bleu_score:
            torch.save(model.state_dict(), config.model_path)
            best_bleu_score = bleu_score

        # if dev_loss < best_dev_loss:
        #     torch.save(model.state_dict(), config.model_path)
        #     best_dev_loss = dev_loss

    # writer.close()


def evaluate(model, data, mode='dev', use_beam=True):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_zh = chinese_tokenizer_load()
    tgt = []
    refs = []
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的中文句子
            zh_sent = batch.tgt_text
            src = batch.src
            src_mask = padding_mask(src, pad_idx=0)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.PAD, config.BOS, config.EOS,
                                               config.beam_size, config.device)
                decode_result = [h[0] for h in decode_result]
            else:
                # decode_result.shape [batch_size, predict_len]
                decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)

            # a = [12907, 277, 7419, 7318, 18384, 28724]
            # sp.decode_ids(a) ,type 是个str类型
            translation = [sp_zh.decode_ids(_s) for _s in decode_result]
            tgt.extend(zh_sent)
            refs.extend(translation)

    if mode == 'test':
        with open(config.output_path, "w", encoding='utf-8') as fp:
            for i in range(len(tgt)):
                line = "idx:" + str(i) + tgt[i] + '|||' + refs[i] + '\n'
                fp.write(line)
    tgt = [tgt]
    bleu = sacrebleu.corpus_bleu(refs, tgt, tokenize='zh')
    return float(bleu.score)


def test(model, test_data, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        # model_par = torch.nn.DataParallel(model)
        model.eval()
        # 开始预测
        test_loss = run_epoch(model, test_data, LossCompute(model.generator, criterion, None))
        # test_loss = run_epoch(model_par, test_data,
        #                       MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(model, test_data, 'test', use_beam=False)
        logger.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def run():

    # 数据预处理
    train_dataset = MTDataset(config.train_file)
    dev_dataset = MTDataset(config.dev_file)
    test_dataset = MTDataset(config.test_file)
    logger.info("-------- Dataset Build! --------")


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size,
                                shuffle=False, collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=False, collate_fn=test_dataset.collate_fn)
    logger.info("-------- Get Dataloader! --------")

    # 初始化模型
    model = make_model(
        config.src_vocab_size,
        config.tgt_vocab_size,
        config.num_layers,
        config.dim_model,
        config.ffn_dim,
        config.num_heads,
        config.dropout
    )

    # model_par = torch.nn.DataParallel(model)
    criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=0, smoothing=0.0)
    optimizer = get_std_opt(model)

    # 训练
    train(model, train_dataloader, dev_dataloader, criterion, optimizer)
    # 测试
    test(model, test_dataloader, criterion)


if __name__ == '__main__':
    run()
