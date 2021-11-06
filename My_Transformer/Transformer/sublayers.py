import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from My_Transformer.Transformer.clone import clones


def attention(query, key, value, mask=None, dropout=None):
    # 输入进来的维度分别是:
    # query: [batch_size, num_heads, len_q, dim_k]
    # key：[batch_size, num_heads, len_k, dim_k]
    # value: [batch_size, num_heads, len_k, dim_v]
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)

    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 输入进来的数据形状：
        # query: [batch_size, len_q, d_model],
        # key: [batch_size, len_k, d_model],
        # value: [batch_size, len_k, d_model]
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        batch_size = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        # query: [batch_size, num_heads, len_q, dim_k]
        # key: [batch_size, num_heads, len_k, dim_k]
        # value:[batch_size, num_heads, len_k, dim_v]
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1、β为全0
        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 按最后一个维度计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 返回Layer Norm的结果

        return self.alpha * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.beta
