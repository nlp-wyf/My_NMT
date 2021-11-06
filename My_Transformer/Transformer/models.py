import copy

import torch.nn as nn
import torch.nn.functional as F

from My_Transformer.Transformer.embedding import Embeddings, PositionalEncoding
from My_Transformer.Transformer.layers import EncoderLayer, DecoderLayer
from My_Transformer.Transformer.sublayers import LayerNorm, MultiHeadedAttention, PositionwiseFeedForward
from My_Transformer.Transformer.clone import clones
from My_Transformer.config import load_config

config = load_config()


class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model).to(config.device)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(config.device)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout).to(config.device)
    # 实例化EncoderLayer、DecoderLayer对象
    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout).to(config.device)
    decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(config.device)
    # 实例化Encoder、Decoder对象
    encoder = Encoder(encoder_layer, N).to(config.device)
    decoder = Decoder(decoder_layer, N).to(config.device)
    # 实例化Embeddings对象
    src_embedding = Embeddings(d_model, src_vocab).to(config.device)
    tgt_embedding = Embeddings(d_model, tgt_vocab).to(config.device)
    # 实例化Generator对象
    generator = Generator(d_model, tgt_vocab).to(config.device)

    # 实例化Transformer模型对象
    model = Transformer(
        encoder,
        decoder,
        nn.Sequential(src_embedding, c(position)),
        nn.Sequential(tgt_embedding, c(position)),
        generator)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(config.device)
