import numpy as np
from operator import itemgetter
import copy

class Variable(object):
    def __init__(self, x):
        if x.dtype != np.float32 and x.dtype != np.int32:
            msg = "Variable content's dtype should be `np.float32` or `np.int32`"
            raise TypeError(msg)

        self.data = x
        self.prev_func = None
        self.depth = 0
        self.grad = None

    def _set_prev_func(self, func):
        self.prev_func = func
        self.depth = func.depth

    def set_grad(self, grad):
        self.grad = grad

    def cleargrad(self):
        self.grad = None

    def get_data(self):
        return self.data

    def shape(self):
        return self.data.shape

    def __len__(self):
        return len(self.data)

    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return "Variable(" + self.data.__repr__() + ")"

    def __str__(self):
        return "Variable(" + self.data.__repr__() + ")"

    def __getitem__(self, item):
        return self

    def backward(self):
        if self.prev_func == None:
            return

        prev_funcs = [self.prev_func]
        variables = [self]
        depth_list = [self.depth]

        while True:
            order = np.argsort(depth_list)
            prev_funcs = sorted(prev_funcs, key=itemgetter(*order))
            variables = sorted(variables, key=itemgetter(*order))
            grads, inputs = prev_funcs[0].backward([variables[0].grad])
            prev_funcs, variables, depth_list = prev_funcs[1:], variables[1:], depth_list[1:]

            for v, g in zip(inputs, grads):
                v.set_grad(g)
                if v.prev_func is None:
                    continue
                prev_funcs.append(v.prev_func)
                variables.append(v)
                depth_list.append(v.depth)

            if prev_funcs == []:
                break
