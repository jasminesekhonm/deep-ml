import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]):
    X  = np.asarray(input_sequence, dtype=float)
    h  = np.asarray(initial_hidden_state, dtype=float)
    Wx = np.asarray(Wx, dtype=float)
    Wh = np.asarray(Wh, dtype=float)
    b  = np.asarray(b, dtype=float)

    for x_t in X:
        h = np.tanh(Wx @ x_t + Wh @ h + b)

    return np.round(h, 4).tolist()