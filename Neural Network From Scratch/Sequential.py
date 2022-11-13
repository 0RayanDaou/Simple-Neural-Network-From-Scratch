from __future__ import print_function
import numpy as np


class Sequential(object):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        if target is not None:
            y = self.loss.forward(y, target)
        return y

    def backward(self):
        y = self.loss.backward()
        for layer in reversed(self.layers):
            y = layer.backward(y)
        return y

    def update_param(self, lr):
        for layer in self.layers:
            layer.update_param(lr)

    def fit(self, x, y, epochs=10, lr=0.1, batch_size=128):
        ret = np.empty(epochs)
        q = int(x.shape[0] / batch_size)

        batch_x = np.empty([q, batch_size, x.shape[1]])
        batch_y = np.empty([q, batch_size, y.shape[1]])
        z = 0
        for i in range(batch_x.shape[0]):
            batch_x[i] = x[z:z + batch_size, :]
            batch_y[i] = y[z:z + batch_size, :]
            z += batch_size
        for epoch in range(epochs):

            temp = np.empty(batch_x.shape[0])
            shuffler = np.random.permutation(len(batch_x))
            batch_x = batch_x[shuffler]
            batch_y = batch_y[shuffler]
            for i in range(batch_x.shape[0]):
                temp[i] = self.forward(batch_x[i], batch_y[i])
                self.backward()
                self.update_param(lr)
            ret[epoch] = np.mean(temp)
        return ret

    def predict(self, x, bs):
        batch_size = bs
        q = int(x.shape[0] / batch_size)
        batch_x = np.empty([q, batch_size, x.shape[1]])
        z = 0
        temp = []
        for i in range(batch_x.shape[0]):
            batch_x[i] = x[z:z + batch_size, :]
            z += batch_size
        for i in range(batch_x.shape[0]):
            temp.append(self.forward(batch_x[i]))
        y = np.array([item for sublist in temp for item in sublist])

        ret = np.empty([y.shape[0]])
        for i in range(y.shape[0]):
            row = y[i]
            max = np.max(row)
            ret[i] = np.where(row == max)[0][0]
        return ret