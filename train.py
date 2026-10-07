import os
import torch
from model.dataset import get_dataloader
from model.transformer import LLMFromScratch

VOCAB_SIZE = 1000
D_MODEL = 128
N_LAYER = 4
N_HEAD = 4
BLOCK_SIZE = 64
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 15
DEVICE = "cpu"

def train():
    print(f"Using device: {DEVICE}")
    
    dataloader = get_dataloader(
        text_path="data/corpus.txt", 
        tokenizer_path="tokenizer/tokenizer.json",
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE
    )
    
    model = LLMFromScratch(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        block_size=BLOCK_SIZE
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training loop...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            print(f"Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}", end="\r")
            
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}] Complete - Avg Loss: {avg_loss:.4f}")
        
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete! Model weights saved to {checkpoint_path}")

if __name__ == "__main__":
    train()