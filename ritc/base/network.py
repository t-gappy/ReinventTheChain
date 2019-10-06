class Network(object):
    def __init__(self):
        self.params = {}

    def param(self):
        for k in self.params.keys():
            yield self.params[k]

    def cleargrads(self):
        for k in self.params.keys():
            self.params[k].cleargrad()


class Chain(Network):
    def __init__(self):
        self.params = {}
        self.networks = []
        return

    def add_net(self, network):
        self.networks.append(network)

    def get_all_param(self):
        for net in self.networks:
            yield net.param()

    def cleargrads(self):
        for net in self.networks:
            net.cleargrads()
