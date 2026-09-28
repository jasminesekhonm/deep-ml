import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(z, epsilon=1e-08):
    z = z - np.max(z, axis=-1, keepdims=True)
    exponentials = np.exp(z)
    return exponentials / (np.sum(exponentials, axis=-1, keepdims=True) + epsilon)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    seq_len = mask.shape[0]
    d_k = Q.shape[-1]
    if K.shape[-1] != d_k:
        raise ValueError("Q and K must share the key dimension")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V must have the same sequence length")
    if mask.shape != (Q.shape[0], K.shape[0]):
        raise ValueError(f"mask must have shape {(Q.shape[0], K.shape[0])}")

    attn_weights = Q @ K.T / np.sqrt(K.shape[-1])
    attn_weights = attn_weights + mask 

    attn_scores = softmax(attn_weights)
    output = attn_scores @ V 
    return np.round(output, 4)