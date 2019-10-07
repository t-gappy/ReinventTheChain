import numpy as np
import copy

class Optimizer(object):
    def __init__(self, lr=1e-2):
        self.lr = lr
        self.network = None

    def setup(self, network):
        self.network = network

    def optimize(self):
        for param in self.network.get_all_param():
            param.data -= self.lr*param.grad
