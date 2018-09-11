import numpy as np
from utils import mse, lin

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y, lr=0.01, n_iter=100):
        # random initialization
        np.random.seed(42)
        self.weights = np.random.rand(X.shape[1], 1)
        self.biases = np.random.rand(1, 1)
        for i in range(n_iter):
            self.gd(X, y, lr)
            if i % 50 == 0:
                y_pred = np.dot(X, self.weights) + self.biases
                print(f'{i} epochs mse {mse(y, y_pred)}')

    def predict(self, X):
        return np.dot(X, self.weights) + self.biases

    def gd(self, X, y, lr):
        y_pred = np.dot(X, self.weights)  + self.biases
        db = 2 * (y_pred - y)
        dw = np.dot(X, db.T).mean(axis=1)
        self.weights -= lr * dw.mean()
        self.biases -= lr * db.mean()
