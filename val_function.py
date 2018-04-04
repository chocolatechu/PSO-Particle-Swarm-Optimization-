import numpy as np


def val_function(x):
    y = x[0]*x[0] + x[1]*x[1] - 10 * np.cos(2 * np.pi * x[0]) - 10 * np.cos(2 * np.pi * x[1]) + 20
    return y
