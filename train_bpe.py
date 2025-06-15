from datasets import load_dataset
from tokenizers import (
    models,
    Tokenizer,
    trainers,
    pre_tokenizers,
)
import numpy as np
from tqdm import tqdm


dataset = load_dataset("muzaffercky/kurdish-kurmanji-news")

train_data = dataset["train"]
test_data = dataset["test"]

np.random.seed(42)

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

def evaluate_tokenizer(tokenizer, dataset, sample_size=500):
    """Evaluate tokenizer on a subset of the dataset"""
    if sample_size > len(dataset):
        sample_size = len(dataset)

    # Select random samples for evaluation
    indices = np.random.choice(len(dataset), size=sample_size, replace=False)
    samples = []

    for idx in indices:
        index = int(idx)  # Convert numpy.int64 to Python int
        item = dataset[index]
        if item.get("content"):
            samples.append(item["content"])

    token_counts = []
    unk_counts = []

    for text in tqdm(samples, desc="Evaluating samples"):
        encoding = tokenizer.encode(text)
        token_counts.append(len(encoding.tokens))
        unk_counts.append(encoding.tokens.count("[UNK]"))

    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    mean_unks = np.mean(unk_counts)
    unk_percentage = (
        (sum(unk_counts) / sum(token_counts)) * 100 if sum(token_counts) > 0 else 0
    )

    return {
        "mean_tokens": mean_tokens,
        "median_tokens": median_tokens,
        "mean_unks": mean_unks,
        "unk_percentage": unk_percentage,
        "token_counts": token_counts,
        "unk_counts": unk_counts,
    }



vocab_sizes = [10000, 12000, 15000, 18000, 20000, 30000, 40000, 50000]
results = {}

for vocab_size in vocab_sizes:
    np.random.seed(42)
    tokenizer = train_bpe_tokenizer(vocab_size, train_data)
    results[vocab_size] = evaluate_tokenizer(tokenizer, test_data)
    print(f"\nResults for vocab_size={vocab_size}:")
    print(f"  Mean tokens per text: {results[vocab_size]['mean_tokens']:.2f}")
    print(f"  Median tokens per text: {results[vocab_size]['median_tokens']:.2f}")
    print(f"  UNK token percentage: {results[vocab_size]['unk_percentage']:.4f}%")
