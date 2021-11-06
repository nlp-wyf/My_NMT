import torch
from torch.autograd import Variable

from My_Transformer.data_loader import sequence_mask


def batch_greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    batch_size, src_seq_len = src.size()
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0

    memory = model.encode(src, src_mask)
    tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data)

    for s in range(max_len):
        # tgt_mask.shape [batch_size, len_tgt, len_tgt]
        tgt_mask = sequence_mask(tgt).expand(batch_size, -1, -1).type_as(src.data)
        # out.shape [batch_size, len_tgt, d_model]
        out = model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask))

        # out[:, -1, :].shape [batch_size, d_model]
        # prob.shape [batch_size, tgt_vocab_size]
        prob = model.generator(out[:, -1, :])
        # pred.shape 为[batch_size]
        pred = torch.argmax(prob, dim=-1)

        # pred.unsqueeze(1).shape [batch_size, 1]
        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
        pred = pred.cpu().numpy()
        for i in range(batch_size):
            # print(stop_flag[i])
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break

    return results


def greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    """传入一个训练好的模型，对指定数据进行预测"""
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(sequence_mask(ys).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        if next_word == end_symbol:
            break
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
