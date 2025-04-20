from datasets import load_dataset
from tokenizers import (
    models,
    Tokenizer,
    trainers,
    pre_tokenizers,
)

dataset = load_dataset("muzaffercky/kurdish-kurmanji-articles", split="train")


def prepare_data(dataset):
    for row in dataset:
        content = row.get("content")
        if content:
            yield content


def train_bpe_tokenizer(vocab_size, dataset):
    corpus_iterator = prepare_data(dataset)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)


    return tokenizer


tokenizer = train_bpe_tokenizer(20000, dataset)
tokenizer.save("tokenizer.json")
