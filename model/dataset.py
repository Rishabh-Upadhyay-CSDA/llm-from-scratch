import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, text_path: str, tokenizer_path: str, block_size: int = 64):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.block_size = block_size
        
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        encoded = self.tokenizer.encode(text)
        self.tokens = encoded.ids
        print(f"Dataset Loaded: {len(self.tokens)} total tokens.")

    def __len__(self):
        # Step in non-overlapping chunks of block_size
        return max(0, (len(self.tokens) - 1) // self.block_size)

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        chunk = self.tokens[start_idx : start_idx + self.block_size + 1]
        
        # Pad chunk if it falls short at the end
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [0] * (self.block_size + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_dataloader(text_path: str, tokenizer_path: str, batch_size: int = 4, block_size: int = 64):
    dataset = TextDataset(text_path, tokenizer_path, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader