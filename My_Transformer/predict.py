import numpy as np
import torch

from My_Transformer.Transformer.beam_search import beam_search
from My_Transformer.Transformer.greedy_search import batch_greedy_decode
from My_Transformer.Transformer.models import make_model
from My_Transformer.config import load_config

from My_Transformer.data_loader import padding_mask
from My_Transformer.utils import chinese_tokenizer_load, english_tokenizer_load

config = load_config()


def translate(model, src, use_beam=True):
    """用训练好的模型进行预测单句，打印模型翻译结果"""
    sp_zh = chinese_tokenizer_load()
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        src_mask = padding_mask(src, pad_idx=0)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.PAD, config.BOS, config.EOS,
                                           config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        translation = [sp_zh.decode_ids(_s) for _s in decode_result]
        print(translation[0])


def one_sentence_translate(sent, beam_search=True):
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
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    translate(model, batch_input, use_beam=beam_search)


def translate_example():
    """单句翻译示例"""
    # sent = "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a " \
    #        "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit " \
    #        "to childless workers."
    sent = "The Not-So-Dire Future of Work"
    # tgt: 近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平，扩大对无子女劳动者的工资所得税减免。
    one_sentence_translate(sent, beam_search=False)


if __name__ == '__main__':
    translate_example()
