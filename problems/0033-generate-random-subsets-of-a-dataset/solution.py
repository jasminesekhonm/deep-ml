import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n_samples, n_features = X.shape 
    random_subsets = []
    if replacements:
        for _ in range(n_subsets):
            idx = np.random.choice(np.arange(n_samples), n_samples, replace=True)
            random_subsets.append((X[idx], y[idx]))
    elif not replacements:
        n_samples_subset = n_samples // 2
        for _ in range(n_subsets):
            idx = np.random.choice(np.arange(n_samples), n_samples_subset, replace=False)
            random_subsets.append((X[idx], y[idx]))
    return random_subsets

        
    