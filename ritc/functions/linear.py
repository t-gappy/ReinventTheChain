from ritc.base import Function
from chainer import cuda

class Linear(Function):
    def __call__(self, x, W, b):
        inputs = [x, W, b]
        outputs = super(Linear, self).__call__(inputs)
        return outputs

    def forward(self, inputs):
        x, W, b = inputs
        y = x.dot(W.T) + b
        return [y]

    def backward(self, grads):
        grad_y = grads[0]
        x, W, _ = self.inputs
        xp = cuda.get_array_module(x.data)
        y = self.outputs[0]

        grad_x = xp.dot(grad_y, W.data)
        grad_W = xp.dot(grad_y.T, x.data)
        grad_b = xp.sum(grad_y, axis=0)

        return [grad_x, grad_W, grad_b], self.inputs


def linear(x, W, b):
    return Linear()(x, W, b)[0]
