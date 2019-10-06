class Optimizer(object):
    def __init__(self, lr=1e-2):
        self.lr = lr
        self.network = None

    def setup(self, network):
        self.network = network

    def optimize(self):
        for network_params in self.network.get_all_param():
            for param in network_params:
                param.data -= self.lr*param.grad
