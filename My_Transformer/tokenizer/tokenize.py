import sentencepiece as spm


def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
    :param input_file: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                       By default, SentencePiece normalizes the input with Unicode NFKC.
                       You can pass a comma-separated list of files.
    :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
    :param model_name: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
                       The input sentence must be pretokenized when using word type.
    :param character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with
                               rich character set like Japanse or Chinese and 1.0 for other languages with
                               small character set.
    """
    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ' \
                     '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)
    spm.SentencePieceTrainer.Train(cmd)


def run():
    en_input = '../parallel_corpus/corpus.en'
    en_vocab_size = 8000
    en_model_name = 'en'
    en_model_type = 'bpe'
    en_character_coverage = 1
    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

    ch_input = '../parallel_corpus/corpus.zh'
    ch_vocab_size = 8000
    ch_model_name = 'zh'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995
    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)


def test():
    sp = spm.SentencePieceProcessor()
    text = "美国总统特朗普今日抵达夏威夷。"

    sp.Load("./zh.model")
    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))
    # a = sp.EncodeAsIds(text)
    a = [286, 69, 295, 4924, 5891, 5330, 6469, 5537, 7573, 5076]
    print(type(sp.decode_ids(a)))
    print('-' * 30)
    pad = sp.pad_id()
    print(pad)
    unk = sp.unk_id()
    print(unk)
    bos = sp.bos_id()
    eos = sp.eos_id()
    print(bos, eos)


if __name__ == "__main__":
    # run()
    test()
