from ritc.base import Function
from chainer import cuda
import numpy as np

class SoftmaxCrossEntropy(Function):
    def __call__(self, x, t):
        inputs = [x, t]
        outputs = super(SoftmaxCrossEntropy, self).__call__(inputs)
        return outputs

    def forward(self, inputs):
        x, t = inputs
        B = x.shape[0]
        xp = cuda.get_array_module(x)

        exp_x = x - xp.max(x)
        exp_x = xp.exp(exp_x)
        exp_x = exp_x / xp.sum(exp_x)

        log_x = xp.log(exp_x+1e-8)
        loss = - xp.sum(log_x[xp.arange(B), t])
        loss /= xp.array(B, dtype=np.float32)
        return [loss]

    def backward(self, grads):
        grad_y = grads[0]
        x, t = self.inputs
        B = x.data.shape[0]
        xp = cuda.get_array_module(x)

        t_onehot = xp.zeros_like(x.data)
        t_onehot[xp.arange(B), t.data] = 1.
        grad_x = (x.data - t_onehot)
        grad_x /= xp.array(B, dtype=np.float32)
        grad_x *= xp.array(grads, dtype=np.float32)

        return [grad_x, None], self.inputs

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)[0]
