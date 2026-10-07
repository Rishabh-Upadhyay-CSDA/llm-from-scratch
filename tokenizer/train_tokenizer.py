import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train_bpe_tokenizer(corpus_path: str, vocab_size: int = 5000, save_path: str = "tokenizer/tokenizer.json"):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Please place your training text there.")

    # Initialize a Byte-Pair Encoding model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Split text on whitespace and byte-level characters
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Configure BPE trainer with target vocabulary size and special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    
    print(f"Training BPE Tokenizer on {corpus_path}...")
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer trained successfully! Saved to: {save_path}")

if __name__ == "__main__":
    # Ensure sample data exists for testing
    os.makedirs("data", exist_ok=True)
    sample_corpus = "data/corpus.txt"
    
    if not os.path.exists(sample_corpus):
        with open(sample_corpus, "w", encoding="utf-8") as f:
            f.write("Large language models are trained using transformers, tokenization, and deep learning.\n" * 100)
            
    train_bpe_tokenizer(corpus_path=sample_corpus, vocab_size=1000)