import torch
import math 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        B, T, d_model = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x) # B, T, d_model 

        Q = Q.view(B, T, self.num_heads, -1)
        K = K.view(B, T, self.num_heads, -1)
        V = V.view(B, T, self.num_heads, -1)

        attention_scores = Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1))

        if mask is not None:
            mask = mask.view_as(attention_scores)
            attention_scores = attention_scores + mask 
        
        attention_weights = F.softmax(attention_scores, dim = -1)
        attention_output = attention_weights @ V
        attention_output =  attention_output.view(B, T, -1)

        return self.out_proj(attention_output)

