import numpy as np

def softmax(z, eps=1e-05):
    z = z - z.max(axis=-1, keepdims=True)
    exponentials = np.exp(z)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    Q_new = x_new @ W_Q 
    K_new = (x_new @ W_K)[np.newaxis, :]    # (1, d_k)
    V_new = (x_new @ W_V)[np.newaxis, :]

    if cache is None:
        cache = (K_new, V_new)
    else:
        K_prev, V_prev = cache 
        cache = (np.vstack([K_prev, K_new]), np.vstack([V_prev, V_new]))
    K_cache, V_cache = cache 
    attn_scores = Q_new @ K_cache.T / np.sqrt(K_cache.shape[-1])

    attn_weights = softmax(attn_scores)
    output = attn_weights @ V_cache
    return (output, cache)
