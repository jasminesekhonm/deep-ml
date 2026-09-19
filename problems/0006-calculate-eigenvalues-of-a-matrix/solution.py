import numpy as np 

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	matrix = np.array(matrix)
	eigenvalues, eigenvectors = np.linalg.eig(matrix)
	return eigenvalues[eigenvalues.argsort()][::-1]