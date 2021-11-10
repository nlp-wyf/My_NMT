import os
import logging
import sentencepiece as spm


def chinese_tokenizer_load():
    sp_zh = spm.SentencePieceProcessor()
    sp_zh.Load('{}.model'.format("./tokenizer/zh"))
    return sp_zh


def english_tokenizer_load():
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load('{}.model'.format("./tokenizer/en"))
    return sp_en
