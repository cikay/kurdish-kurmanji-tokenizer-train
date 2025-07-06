from datasets import load_dataset
from tokenizers import (
    models,
    Tokenizer,
    trainers,
    pre_tokenizers,
    normalizers,
    processors,
)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import unicodedata
import regex
from tokenizers import PreTokenizedString, NormalizedString, Regex
from tokenizers.pre_tokenizers import Split


def load_and_prepare_dataset():
    """Load the Kurdish dataset"""
    dataset = load_dataset("muzaffercky/kurdish-kurmanji-news")
    return dataset["train"], dataset["test"]


def prepare_data(dataset):
    for row in dataset:
        content = row.get("content")
        yield content.strip() or ""


def regex_split(text: NormalizedString):
    str_text = str(text)
    results = []
    for item in pattern.finditer(str_text):
        token = item.group()
        span = item.span()
        results.append(NormalizedString(token))
    return results


class KurdishRegexPreTokenizer:
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(lambda _, text: regex_split(text))


def train_enhanced_bpe_tokenizer(vocab_size, dataset):
    pattern = Regex(
        """'(?i:ê|yê|yî|an|în|ên|ya|yan)| ?\p{L}+| ?\p{N}{1,3}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    corpus_iterator = prepare_data(dataset)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(pattern, behavior="isolated")

    # Add composed normalization to handle Kurdish-specific characters consistently
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFC()
        ]
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    return tokenizer


def comprehensive_evaluation(tokenizer, dataset, sample_size=1000):
    """Enhanced evaluation with more metrics"""
    if sample_size > len(dataset):
        sample_size = len(dataset)

    # Select random samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), size=sample_size, replace=False)
    samples = []

    for idx in indices:
        item = dataset[int(idx)]
        if item.get("content") and len(item["content"].strip()) > 10:
            samples.append(item["content"].strip())

    # Evaluation metrics
    token_counts = []
    unk_counts = []
    char_counts = []
    compression_ratios = []

    for text in tqdm(samples, desc="Evaluating samples"):
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens

        token_count = len(tokens)
        unk_count = tokens.count("[UNK]")
        char_count = len(text)

        token_counts.append(token_count)
        unk_counts.append(unk_count)
        char_counts.append(char_count)

        # Compression ratio (characters per token)
        if token_count > 0:
            compression_ratios.append(char_count / token_count)

    # Calculate statistics
    results = {
        "mean_tokens": np.mean(token_counts),
        "median_tokens": np.median(token_counts),
        "std_tokens": np.std(token_counts),
        "mean_chars_per_token": np.mean(compression_ratios),
        "median_chars_per_token": np.median(compression_ratios),
        "unk_percentage": (
            (sum(unk_counts) / sum(token_counts)) * 100 if sum(token_counts) > 0 else 0
        ),
        "mean_unks_per_text": np.mean(unk_counts),
        "total_texts": len(samples),
        "vocab_efficiency": np.mean(compression_ratios),  # Higher is better
    }

    return results


def plot_results(results_dict):
    """Create visualization of results"""
    vocab_sizes = list(results_dict.keys())
    mean_tokens = [results_dict[vs]["mean_tokens"] for vs in vocab_sizes]
    chars_per_token = [results_dict[vs]["mean_chars_per_token"] for vs in vocab_sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Mean tokens per text
    ax1.plot(vocab_sizes, mean_tokens, marker="o", linewidth=2, markersize=6)
    ax1.set_xlabel("Vocabulary Size")
    ax1.set_ylabel("Mean Tokens per Text")
    ax1.set_title("Tokenization Efficiency")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Characters per token (compression efficiency)
    ax2.plot(
        vocab_sizes,
        chars_per_token,
        marker="s",
        color="green",
        linewidth=2,
        markersize=6,
    )
    ax2.set_xlabel("Vocabulary Size")
    ax2.set_ylabel("Characters per Token")
    ax2.set_title("Compression Efficiency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kurdish_tokenizer_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main training pipeline"""
    print("Loading Kurdish Kurmanji dataset...")
    train_data, test_data = load_and_prepare_dataset()

    print(
        f"Dataset size: {len(train_data)} training texts, {len(test_data)} test texts"
    )

    # Test different vocabulary sizes
    results = {}

    print("\nTraining tokenizers with different vocabulary sizes...")

    for factor in range(7, 11):
        vocab_size = factor * 2000
        print(f"\n{'='*50}")
        print(f"Training with vocab_size = {vocab_size}")
        print(f"{'='*50}")

        # Train tokenizer
        tokenizer = train_enhanced_bpe_tokenizer(vocab_size, train_data)

        # Evaluate
        results[vocab_size] = comprehensive_evaluation(
            tokenizer, test_data, sample_size=1000
        )

        # Print results
        r = results[vocab_size]
        print(f"\nResults for vocab_size={vocab_size}:")
        print(
            f"  Mean tokens per text: {r['mean_tokens']:.2f} (±{r['std_tokens']:.2f})"
        )
        print(f"  Median tokens per text: {r['median_tokens']:.2f}")
        print(f"  Characters per token: {r['mean_chars_per_token']:.2f}")
        print(f"  UNK token percentage: {r['unk_percentage']:.4f}%")
        print(f"  Vocabulary efficiency: {r['vocab_efficiency']:.2f}")

        # Save the tokenizer for the recommended sizes
        tokenizer.save(f"./tokenizers/{vocab_size}.json")
        print(f"  Saved tokenizer to kurdish_kurmanji_tokenizer_{vocab_size}.json")

    # Save results
    with open("tokenizer_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    plot_results(results)

    return results


if __name__ == "__main__":
    results = main()
