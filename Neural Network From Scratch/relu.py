import numpy as np


class ReluLayer(object):
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = x * (x > 0)
        return self.y

    def backward(self, y_grad):
        back = y_grad * (self.y > 0)
        return back

    def update_param(self, lr):
        pass  # no parameters to update
