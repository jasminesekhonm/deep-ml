import torch

def func(x_value: float):
    return x_value**2 + 3*x_value + 2

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    y = func(x_value)
    dy_dx = 2*x_value + 3 
    return dy_dx 
