import numpy as np

def calculate_correlation_matrix(X, Y=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float) if Y is not None else X 

    n_samples, n_features = X.shape 

    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    cov = Xc.T @ Yc

    norms = np.outer(np.linalg.norm(Xc, axis=0), np.linalg.norm(Yc, axis=0))

    corr = cov / norms if np.all(norms != 0) else 0 

    return np.clip(corr, -1, 1)