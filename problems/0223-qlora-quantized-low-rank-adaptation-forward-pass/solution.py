import numpy as np

def qlora_forward(
	x: list[list[float]],
	quantized_W: list[list[int]],
	scale: float,
	zero_point: float,
	A: list[list[float]],
	B: list[list[float]],
	alpha: float = 1.0
) -> list[list[float]]:
    """
	QLoRA forward pass with 4-bit quantized frozen weights.
	
	Args:
		x: Input matrix (batch_size x in_features)
		quantized_W: 4-bit quantized weights (in_features x out_features)
		             Values are integers that need to be dequantized
		scale: Quantization scale factor
		zero_point: Quantization zero point for dequantization
		A: LoRA matrix A (rank x out_features) - full precision
		B: LoRA matrix B (in_features x rank) - full precision
		alpha: LoRA scaling factor
		
	Returns:
		Output matrix (batch_size x out_features)
	"""
    x, quantized_W, A, B = (np.asarray(m, dtype=float) for m in (x, quantized_W, A, B))
    dequantized_W = quantized_W * scale + zero_point 
    x_1 = x @ dequantized_W
    x_2 = (x @ B) @ A 
    x_2 = x_2 * alpha / B.shape[1]
    return x_1 + x_2 