import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train_bpe_tokenizer(corpus_path: str, output_path: str, vocab_size: int = 1000):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Please add raw text data!")

    # 1. Initialize BPE Model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # 2. Use ByteLevel pre-tokenizer to ensure subwords & fallback byte coverage
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    # 3. Configure Trainer with special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print(f"Training Byte-Level BPE Tokenizer on {corpus_path}...")
    tokenizer.train(files=[corpus_path], trainer=trainer)

    # 4. Save tokenizer JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer trained successfully! Saved to {output_path}")

if __name__ == "__main__":
    train_bpe_tokenizer(
        corpus_path="data/corpus.txt",
        output_path="tokenizer/tokenizer.json",
        vocab_size=1000
    )