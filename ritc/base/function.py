import numpy as np
import copy
from ritc.base.variable import Variable

class Function(object):
    def __call__(self, args):
        input_variables = [arg if isinstance(arg, Variable)
                           else Variable(arg)
                           for arg in args]
        input_arrays = [x.get_data()
                        for x in input_variables]

        output_arrays = self.forward(input_arrays)

        output_variables = [Variable(x)
                            for x in output_arrays]

        self.depth = max([v.depth for v in input_variables]) + 1
        self.inputs = input_variables
        self.outputs = copy.deepcopy(output_arrays)
        for v in output_variables:
            v._set_prev_func(self)

        return output_variables
    
    def __getitem__(self, item):
        return self

    def forward(self, x):
        return

    def backward(self, x):
        return
