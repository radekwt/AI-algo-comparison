import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.0001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        return np.insert(x, 0, 1, axis=1)

    def _transform_y(self, y):
        return np.array([1 if label == 1 else 0 for label in y])

    def fit(self, x, y, max_iter):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for _ in range(max_iter):
            linear_model = np.dot(x, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            gradients_w, gradient_b = self.compute_gradients(x, y, y_pred)
            self.weights -= self.lr * gradients_w
            self.bias -= self.lr * gradient_b

    def predict(self, x):
        x = self._transform_x(x)
        linear_model = np.dot(x, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return np.array([1 if pred > 0.5 else 0 for pred in y_pred])

    def compute_gradients(self, x, y_true, y_pred):
        error = y_pred - y_true
        gradients_w = np.dot(x.T, error) / len(y_true)
        gradient_b = np.mean(error)
        return gradients_w, gradient_b