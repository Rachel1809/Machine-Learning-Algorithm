import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # init parameters
        # X: (samples, features)
        samples, features = X.shape
        
        # Weight: (features, 1)
        self.weight = np.zeros(features)
        
        # Bias: (1, 1)
        self.bias = 0
        
        for _ in range(self.epochs):
            # y_pred: (samples, 1)
            y_predicted = np.dot(X, self.weight) + self.bias
            
            #dw: (features, 1)
            dw = (1 / samples) * np.dot(X.T, (y_predicted - y))
            
            #db: (1, 1)
            db = (1 / samples) * np.sum(y_predicted - y)
            
            self.weight -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weight) + self.bias