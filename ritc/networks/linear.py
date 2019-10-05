from ritc.functions import linear
from ritc.base import Variable
import numpy as np
from chainer import cuda

class Linear(object):
    def __init__(self, input_dim, output_dim, initializer=None):
        self.params = {
            "W": Variable(np.random.normal(0, 1, (output_dim, input_dim)).astype(np.float32)),
            "b": Variable(np.zeros(output_dim).astype(np.float32))
        }

    def __call__(self, x):
        y = linear(x, self.params["W"], self.params["b"])
        return y
