from ritc.base import Function
from chainer import cuda

class ReLU(Function):
    def __call__(self, x):
        inputs = [x]
        outputs = super(ReLU, self).__call__(inputs)
        return outputs

    def forward(self, inputs):
        x = inputs[0]
        xp = cuda.get_array_module(x)

        self.mask = x > 0
        y = xp.maximum(x, 0)
        return [y]

    def backward(self, grads):
        grad_y = grads[0]
        grad_y *= self.mask
        return [grad_y], self.inputs

def relu(x):
    return ReLU()(x)[0]
