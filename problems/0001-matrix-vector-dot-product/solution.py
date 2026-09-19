import numpy as np 

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	# m x n 
	# n x 1 
	a, b = np.array(a), np.array(b)
	if a is None or b is None or a.shape[1] != b.shape[0]:
		return -1 
	return np.dot(a, b)