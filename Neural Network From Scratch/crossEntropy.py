import numpy as np


class CrossEntropyLayer(object):
    def __init__(self):
        self.x = None
        self.t = None

    def forward(self, x, t):
        self.x = x
        self.t = t
        q = (-1/(x.shape[0]))*np.sum(np.log(x)*t)
        q = np.nan_to_num(q)
        return q

    def backward(self, y_grad=None):
        q = (-1/(self.x.shape[0]))*np.divide(self.t, self.x)
        q = np.nan_to_num(q)
        return q
