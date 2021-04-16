import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import mnist

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
        self.__random_seed = 1

        self.history = {}
        self.train_params = {}
        self.data = {}

    def sigmoid(self, t): return 1 / (1 + np.exp(-t))

    def sigmoid_derivative(self, Z):
        p = 1/ (1 + np.exp(-Z))
        return p * (1 - p)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def cross_entropy_loss(self, y, y1):
        return -np.mean(y*np.log(y1+1e-8))

    def init_params(self, rand=False, seed=1):
        params = {}
        self.__random_seed = seed
        np.random.seed(seed)

        if rand:
            for i in range(1, self.__L):
                params['W' + str(i)] = np.random.randn(self.layers_size[i], self.layers_size[i - 1])
                params['b' + str(i)] = np.random.randn(self.layers_size[i], 1)
        else:
            for i in range(1, self.__L):
                params['W' + str(i)] = np.zeros((self.layers_size[i], self.layers_size[i - 1]))
                params['b' + str(i)] = np.zeros((self.layers_size[i], 1))
        return params

    def forward(self, X):
        self.layers["input"] = X.T
        for l in range(1, self.__L):
            if l == 1:
                self.layers["W" + str(l)] = self.sigmoid(np.dot(self.weights["W" + str(l)], self.layers["input"]) + self.weights["b" + str(l)])
                #self.layers["W" + str(l)] = self.sigmoid(np.dot(self.layers["input"], self.weights["W" + str(l)]) + self.weights["b" + str(l)])
            else:
                self.layers["W" + str(l)] = self.sigmoid(np.dot(self.weights["W" + str(l)], self.layers["W" + str(l-1)]) \
                                                        + self.weights["b" + str(l)])
        # last forward step - softmax activation
        self.layers["output"] = self.softmax(np.dot(self.weights["W" + str(self.__L)], self.layers["W" + str(self.__L-1)]) \
                                                        + self.weights["b" + str(self.__L)])
        return self.layers["output"]

    def backward(self, Y):
        d_weights = {}

        # compute
        dZ = self.layers["output"] - Y
        dW = np.dot(dZ, self.layers["W" + str(self.__L - 1)]) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n

        dWPrev = self.weights["W" + str(self.__L)]

        d_weights["W" + str(self.__L)] = dW
        d_weights["b" + str(self.__L)] = db

        for l in range(self.__L-1, 0, -1):
            dZ = np.dot(self.weights["W" + str(l+1)], dZ) \
                    * self.sigmoid_derivative(self.layers["W" + str(l)])
            dW = 1. / self.n * np.dot(dZ, np.dot(self.layers["W" + str(l-1)], self.weights["W" + str(l)]) \
                                                        + self.weights["b" + str(l)])
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)

            d_weights["W" + str(l)] = dW
            d_weights["b" + str(l)] = db

        # update
        for l in range(self.__L):
            self.weights["W" + str(l)] = self.weights["W" + str(l)] - self.train_params["lr"] * d_weights["W" + str(l)]
            self.weights["b" + str(l)] = self.weights["b" + str(l)] - self.train_params["lr"] * d_weights["b" + str(l)]

    def fit(self, x, y, lr = 0.01, n_iters = 2000):
        self.train_params["lr"] = lr
        self.train_params["epochs"] = n_iters
        self.data["X"] = x
        self.data["Y"] = y
        self.n = self.data["X"].shape[0]
        self.history["loss"] = []
        self.history["acc"] = []
        self.layers_size.insert(0, self.data["X"].shape[1])
        self.weights = self.init_params(rand=True)

        for epoch in range(self.train_params["epochs"]):
            self.forward(self.data["X"])
            cost = self.cross_entropy_loss(self.data["Y"], self.layers["output"])
            if epoch%10 == 0:
                self.history["loss"].append(cost)
            if epoch%100 == 0:
                pred = self.predict(self.data["X"], self.data["Y"])
                self.history["acc"].append(pred)
                print("CrossEntropyLoss: ", cost, "train Accuracy: ", pred)

    def predict(self, X, Y):
        output = self.forward(X)
        y_out = np.argmax(output, axis=0)
        y = np.argmax(Y, axis=1)
        acc = (y_out == y).mean()
        return acc

    def plot_loss(self):
        plt.plot(np.arange(len(self.history["loss"])), self.history["loss"])
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()

    def clean_memory(self):
        del self.data
        del self.layers

def pre_process_data(train_x, train_y, test_x, test_y):
    train_x = train_x.reshape((train_x.shape[0], 784))
    test_x = test_x.reshape((test_x.shape[0], 784))
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))

    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    data = mnist.load_data()
    train_x, train_y, test_x, test_y = data[0][0], data[0][1], data[1][0], data[1][1]

    train_x, train_y, test_x, test_y = pre_process_data(train_x, train_y, test_x, test_y)

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    layers_dims = [50, 10]

    snn = SimpleNN(layers_dims)
    snn.fit(train_x, train_y, lr=0.1, n_iters=1000)
    print("Train Accuracy:", snn.predict(train_x, train_y))
    print("Test Accuracy:", snn.predict(test_x, test_y))
    snn.plot_loss()