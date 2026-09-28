import numpy as np

def softmax(z, eps):
    z = z - z.max(axis=-1, keepdims=True)
    exponentials = np.exp(z)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

def multihead_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, num_heads: int, eps: float=1e-5) -> np.ndarray:
    head_dim = q.shape[-1] // num_heads

    output = []
    i = 0
    for _ in range(num_heads):
        q_i = q[:, :, i:i+head_dim]
        k_i = k[:, :, i:i+head_dim]
        v_i = v[:, :, i:i+head_dim]
        attn_scores = q_i @ k_i.swapaxes(1, 2) / np.sqrt(k_i.shape[-1])
        attn_scores = softmax(attn_scores, eps)
        head_output = attn_scores @ v_i 
        output.append(head_output)
        i = i + head_dim
    output = np.concatenate(output, axis=-1)
    return output 

def layer_norm(x: np.ndarray, gamma, beta, eps) -> np.ndarray:
    x_mean = x.mean(axis=-1, keepdims=True)
    x_var = x.var(axis=-1,  keepdims=True)
    return gamma * (x - x_mean) / np.sqrt(x_var + eps) + beta 

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = X.shape 
    W_q, W_k, W_v = weights["W_q"], weights["W_k"], weights["W_v"]
    W_o = weights["W_o"]
    W_1, b_1 = weights["W1"], weights["b1"]
    W_2, b_2 = weights["W2"], weights["b2"]
    gamma1, beta1 = weights["gamma1"], weights["beta1"]
    gamma2, beta2 = weights["gamma2"], weights["beta2"]

    q = X @ W_q 
    k = X @ W_k 
    v = X @ W_v 

    attn_output = multihead_attention(q, k, v, num_heads, eps) @ W_o
    x = X + attn_output
    x = layer_norm(x, gamma1, beta1, eps)
    x_res = x 
    x = x @ W_1 + b_1 
    x = np.where(x >= 0, x, 0)
    x = x @ W_2 + b_2 
    x = x + x_res 
    x = layer_norm(x, gamma2, beta2, eps)

    return x 