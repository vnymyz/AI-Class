"""XOR dataset — classic example a single neuron can't solve."""
import numpy as np

def get_xor_data():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=np.float64)

    y = np.array([
        [0],
        [1],
        [1],
        [0],
    ], dtype=np.float64)

    return X, y
