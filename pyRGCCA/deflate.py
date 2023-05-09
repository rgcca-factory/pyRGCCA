import numpy as np

def deflate(X, w):
    y = X @ w
    beta =  y.T @ X / np.dot(y, y)
    X_defl = X - np.outer(y, beta)
    return X_defl, beta
