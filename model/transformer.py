import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8/3 * d_model)  # Standard Llama expansion scaling
            
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # Up projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadCausalAttention(nn.Module):
    """Multi-Head Causal Self-Attention with Scaled Dot-Product Attention."""
    def __init__(self, d_model: int, n_head: int, block_size: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Combined Linear layer for Q, K, V projections
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Causal mask buffer
        self.register_buffer(
            "tril", 
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size() # Batch, Seq Len, Embedding Dim
        
        # Project and split into Q, K, V
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # PyTorch Optimized Scaled Dot-Product Attention (FlashAttention compatible)
        # Applies causal mask automatically when is_causal=True
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class DecoderBlock(nn.Module):
    """Transformer Decoder Block with Pre-Normalization and Residuals."""
    def __init__(self, d_model: int, n_head: int, block_size: int):
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_head, block_size)
        
        self.ffn_norm = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.attention_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x

class LLMFromScratch(nn.Module):
    """Complete Decoder-Only Language Model."""
    def __init__(self, vocab_size: int, d_model: int = 256, n_layer: int = 6, n_head: int = 8, block_size: int = 128):
        super().__init__()
        self.block_size = block_size
        
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(block_size, d_model)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_head, block_size) for _ in range(n_layer)
        ])
        
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and LM head
        self.tok_embeddings.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.tok_embeddings(idx) + self.pos_embeddings(pos)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            logits = self.lm_head(x[:, -1, :])  # Return logits for the last token only
            return logits