from ritc.functions import linear
from ritc.base import Variable
from ritc.base import Network
import numpy as np
from chainer import cuda

class Linear(Network):
    def __init__(self, input_dim, output_dim, initializer=None):
        super(Linear, self).__init__()
        W = Variable(
            np.random.normal(
                0,
                (2/input_dim)**0.5,
                (output_dim, input_dim)
            ).astype(np.float32)
        )
        b = Variable(np.zeros(output_dim).astype(np.float32))

        self.params["W"] = W
        self.params["b"] = b

    def __call__(self, x):
        y = linear(x, self.params["W"], self.params["b"])
        return y
