import numpy as np


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def sigmoid_derivative(p):
    return p * (p - 1)


class SimpleNN:
    def __init__(self, layers, nClasses=2):
        self.layers = layers
        self.parameters = self.init_params()



    def init_params(self):
        params = {}
        for i in range(1, len(self.layers)):
            params['W' + str(i)] = np.zeros(self.layers[i], self.layers[i - 1])
            params['b' + str(i)] = np.zeros(self.layers[i], 1)
        return params

    def forward(self):
        return 0

    def backward(self):
        return 0

    def fit(self, x, y):