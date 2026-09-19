import numpy as np 
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	matrix = np.array(matrix)
	if mode not in ['row', 'column']:
		return []
	axis_ = (1 if mode == 'row' else 0)
	return np.mean(matrix, axis=axis_)