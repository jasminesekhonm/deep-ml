import numpy as np

def lora_forward(
	x: list[list[float]],
	W: list[list[float]],
	A: list[list[float]],
	B: list[list[float]],
	alpha: float = 1.0
) -> list[list[float]]:
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    batch_size, in_features = x.shape 
    x_1 = x @ W # batch_size x out_features 
    x_2 = (x @ B) @ A # batch_size x out_features 
    x_2 = x_2 * alpha / B.shape[1]
    return (x_1 + x_2 ).tolist()