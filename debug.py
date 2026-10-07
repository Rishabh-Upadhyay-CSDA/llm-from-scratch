from tokenizers import Tokenizer

# Load tokenizer from local JSON file
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# Retrieve the full vocabulary dictionary: {token_string: token_id}
vocab = tokenizer.get_vocab()

print(f"Total Vocabulary Size: {len(vocab)}\n")

# Sort by Token ID (0 to Vocab Size - 1)
sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

# Print the first 50 token mappings
print("--- First 50 Tokens ---")
for token, token_id in sorted_vocab[:50]:
    print(f"ID {token_id:5d} -> '{token}'")

# Print special token IDs if available
print("\n--- Special Tokens ---")
for token_id in range(min(10, len(vocab))):
    raw_token = tokenizer.id_to_token(token_id)
    print(f"ID {token_id:2d}: {raw_token}")