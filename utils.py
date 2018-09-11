import numpy as np

def mse(y_true, y_pred):                  
    return np.mean((y_true - y_pred) ** 2)

def lin(w, b, x):
    return np.dot(x, w) + b
