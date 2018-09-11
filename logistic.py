import numpy as np

class LogisticRegression:
    def __init__(self):
        pass

    def fit(self, X, y, lr=0.01, n_epochs=100):
        self.weights = np.random.rand(1, X.shape[1])
        self.biases = np.random.rand(1, 1)
        for i in range(n_epochs):
            self.gd(X, y, lr)
            if i % 100 == 0:
                y_pred = self.predict_proba(X) 
                print(f'weight {self.weights[0, 0]}')
                print(f'biases {self.biases[0, 0]}')
                print(f'{i} epochs cross_entropy {self.cross_entropy(y, y_pred)}')

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.biases
        return self.sigmoid(z)

    def predict(self, X):
        a = self.predict_proba(X)  
        return np.where(a>.5, 1, 0)

    def gd(self, X, y, lr=0.01):
        y_pred = self.predict_proba(X)
        db = y_pred - y
        dw = np.dot(X.T, y_pred - y)
        self.weights -= lr * dw.mean()
        self.biases -= lr * db.mean()

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def cross_entropy(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
