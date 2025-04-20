from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "muzaffercky/kurdish-kurmanji-tokenizer"
)

ids = tokenizer.encode("Ez diçim malê")
tokens = tokenizer.tokenize("Ez diçim malê")


print(f"Tokens: {tokens}")
print(f"IDs: {ids}")
