import numpy as np
from scipy.stats import mode

class KNN:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_trn = X
        self.y_trn = y

    def compute_dist(self, X):
        """
            (A - B)2 = A2 + B2 - 2AB 
        """
        # calculate squared
        a2 = np.power(X, 2)
        b2 = np.power(self.X_trn, 2)

        # add empty dimensions for broadcast sum
        a2_b2 = a2[:, np.newaxis, :] + b2[np.newaxis, :]
        a2_b2 = a2_b2.sum(axis=2)
        ab = X.dot(self.X_trn.T)
        dists = a2_b2 - 2 * ab
        return dists

    def predict(self, X, k=1):
        # compute distance & select top k
        dists = self.compute_dist(X)
        args = np.argsort(dists, axis=1)[:, :k]
        labels = self.y_trn[args]
        
        # select most frequent
        m = mode(labels, axis=1)[0]
        preds = np.array([o[0] for o in m]) 
                
        return preds

