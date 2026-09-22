import numpy as np

def forward_pass(X, weights):
    return np.dot(X, weights)

def mse_loss(y, y_pred):
    return np.mean((y_pred - y)**2, axis=0)
       
def one_iteration(X, y, weights, learning_rate):
    n_samples = X.shape[0]
    y_pred = forward_pass(X, weights)
    error = mse_loss(y, y_pred) 
    dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
    updated_weights = weights - learning_rate * dw 
    return error, updated_weights 

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    n_samples, n_features = X.shape 

    if method == "batch":
        for i in range(n_epochs):
            error, updated_weights = one_iteration(X, y, weights, learning_rate)
            weights = updated_weights

    elif method == "stochastic":
        for i in range(n_epochs):
            for j in range(n_samples):
                x_ = X[j, :][None, :]
                y_ = y[j:j+1]
                error, updated_weights = one_iteration(x_, y_, weights, learning_rate)
                weights = updated_weights
    
    elif method == "mini_batch":
        if batch_size is None or batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        
        for i in range(n_epochs):
            for j in range(0, n_samples, batch_size):
                x_b = X[j:j+batch_size, :]
                y_b = y[j:j+batch_size]
                error, updated_weights = one_iteration(x_b, y_b, weights, learning_rate)
                weights = updated_weights
    
    return weights 






    