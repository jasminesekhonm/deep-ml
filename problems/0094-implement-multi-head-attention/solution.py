import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    return (X @ W_q, X @ W_k, X @ W_v)

def softmax(z, eps=1e-05):
    z = z - z.max(axis=-1, keepdims=True)
    exponentials = np.exp(z)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    
def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    attn_scores = Q @ K.T / np.sqrt(K.shape[-1])
    attn_weights = softmax(attn_scores)

    return attn_weights @ V 

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    
    d_model = Q.shape[-1]
    head_dim = d_model // n_heads 

    outputs = []
    i = 0 
    for _ in range(n_heads):
        q_i = Q[:, i:i+head_dim]
        k_i = K[:, i:i+head_dim]
        v_i = V[:, i:i+head_dim]

        output_i = self_attention(q_i, k_i, v_i)
        outputs.append(output_i)
        i = i + head_dim
    
    outputs = np.concatenate(outputs, axis=-1)
    return outputs


