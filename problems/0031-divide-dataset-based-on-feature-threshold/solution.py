import numpy as np 

def divide_on_feature(X, feature_i, threshold):
    n_samples, n_features = X.shape 
    if feature_i >= n_features:
        raise ValueError('Feature Index should be less than or equal to number of features in X')
    
    X_meets_cond = X[X[:, feature_i] >= threshold]
    X_other = X[X[:, feature_i] < threshold]
    return [X_meets_cond, X_other]