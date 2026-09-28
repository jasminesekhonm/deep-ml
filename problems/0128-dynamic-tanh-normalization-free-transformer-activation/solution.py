import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    x = alpha * x 
    x = np.tanh(x)
    x = gamma * x + beta 
    return x