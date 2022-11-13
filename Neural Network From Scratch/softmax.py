import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        self.y = None

    def forward(self, x):
        f = np.ndarray(x.shape)
        temp = np.ndarray(x.shape)
        q, l = x.shape
        for i in range(q):
            for j in range(l):
                temp[i][j] = x[i][j] - np.max(x[i])
            for k in range(l):
                f[i][k] = np.exp(temp[i][k]) / np.sum(np.exp(temp[i]))
        self.y = f
        return self.y

    def backward(self, y_grad):
        b = np.ndarray(y_grad.shape)
        n, m = self.y.shape
        for i in range(n):
            i_grad = np.diag(self.y[i]) - self.y[i].reshape(m, 1).dot(self.y[i].reshape(1, m))
            b[i] = y_grad[i].dot(i_grad)
        return b

    def update_param(self, lr):
        pass  # no learning for softmax layer
