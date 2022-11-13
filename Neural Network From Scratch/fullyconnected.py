import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        self.x = None
        self.W_grad = None
        self.b_grad = None

        # need to initialize self.W and self.b
        self.W = np.random.uniform(low=0, high=np.sqrt(2/(n_o+n_i)), size=(n_o, n_i))

        self.b = np.zeros((1, n_o))

    def forward(self, x):
        self.x = x
        f = np.add(np.matmul(self.x, np.transpose(self.W)), self.b)
        return f

    def backward(self, y_grad):
        self.W_grad = y_grad.T.dot(self.x)
        self.b_grad = y_grad
        f = y_grad.dot(self.W)
        return f

    def update_param(self, lr):
        self.W = self.W - lr*self.W_grad
        self.b = self.b - lr*self.b_grad
