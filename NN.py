import numpy as np


class SimpleNN:
    def __init__(self, layers_list, type='classification'):
        """
        :param layers_list:
        :param type: classification or regression (last layer softmax or sigmoid)
        __L: count of layers
        """
        self.layers_size = layers_list
        self.type = type
        self.layers = {}
        self.weights = {}

        self.__L = len(layers_list)
        self.n = 0
        self.costs = []
        self.__random_seed = 1

    def sigmoid(self, t): return 1 / (1 + np.exp(-t))

    def sigmoid_derivative(self, p): return p * (p - 1)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def cross_entropy_loss(self, y, y1):
        return -np.mean(y*np.log(y1+1e-8))

    def init_params(self, rand=False, seed=42):
        params = {}
        self.__random_seed = seed
        np.random.seed(seed)

        if rand:
            for i in range(1, self.__L):
                params['W' + str(i)] = np.random.randn(self.layers_size[i], self.layers_size[i - 1])
                params['b' + str(i)] = np.random.randn(self.layers_size[i], 1)
        else:
            for i in range(1, self.__L):
                params['W' + str(i)] = np.zeros(self.layers_size[i], self.layers_size[i - 1])
                params['b' + str(i)] = np.zeros(self.layers_size[i], 1)
        return params

    def forward(self, X):
        self.layers["input"] = X
        for l in range(1, self.__L-1):
            if l == 1:
                self.layers["W" + str(l)] = self.sigmoid(np.dot(self.layers["input"], self.weights["W" + str(l)]) \
                                                        + self.weights["b" + str(l)])
            else:
                self.layers["W" + str(l)] = self.sigmoid(np.dot(self.layers["W" + str(l-1)], self.weights["W" + str(l)]) \
                                                        + self.weights["b" + str(l)])
        # last forward step - softmax activation
        self.layers["output"] = self.softmax(np.dot(self.layers["W" + str(self.__L-1)], self.weights["W" + str(self.__L)]) \
                                                        + self.weights["b" + str(self.__L)])

    def backward(self, Y):
        d_weights = {}
        dZ = self.layers["output"] - Y

    def fit(self, x, y):
        return 0