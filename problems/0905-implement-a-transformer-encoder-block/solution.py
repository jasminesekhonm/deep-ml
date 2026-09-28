import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        x_in = x 
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout(x)
        x = x + x_in 
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x 
