def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)

    x = x0 
    for iter in range(1,num_iterations+1):
        # y = f(x)
        gradients = grad(x)
        m = beta1*m + (1-beta1)*gradients
        v = beta2*v + (1-beta2)*gradients**2 
        m_hat = m / (1 - beta1**iter)
        v_hat = v / (1 - beta2**iter)
        x = x - learning_rate * m_hat / (np.sqrt(v_hat)+epsilon)


    return x 