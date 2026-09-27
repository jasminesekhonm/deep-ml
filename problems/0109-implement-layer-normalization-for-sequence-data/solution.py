import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    batch_size, seq_length, n_features = X.shape 
    if beta.shape[-1] != n_features or gamma.shape[-1]!=n_features:
        raise ValueError("dimensions mismatch")
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    X = (X -  mean)/(np.sqrt(var) + epsilon)
    
    return gamma * X + beta 