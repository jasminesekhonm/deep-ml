import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def softmax(scores, epsilon=1e-05):
    seq_len = scores.shape[0]
    exp_ = np.exp(scores)
    denom = np.sum(exp_, axis=1, keepdims=True) # + epsilon
    return exp_ / denom 

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    seq_len, d_k = Q.shape 
    seq_len, d_v = V.shape 
    attn_scores = Q @ K.T / np.sqrt(d_k)
    attn_weights = softmax(attn_scores)
    return attn_weights @ V 
