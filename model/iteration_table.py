class IterationTable():

    def __init__(self):
        # initialize table
        self.iteration_table = {}

        iteration = 13
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=10.0, edge_weight=5.0, lpip_weight=0.0, is_bce=0)

        iteration = 14
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=5.0, edge_weight=10.0, lpip_weight=0.0, is_bce=0)

        iteration = 15
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=10.0, edge_weight=10.0, lpip_weight=0.0, is_bce=0)

        iteration = 16
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=10.0, edge_weight=0.0, lpip_weight=0.0, is_bce=0)

        iteration = 17
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=0.0, edge_weight=10.0, lpip_weight=0.0, is_bce=0)

        iteration = 18
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=10.0, edge_weight=5.0, lpip_weight=1.0, is_bce=0)

        iteration = 19
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=5.0, edge_weight=10.0, lpip_weight=1.0, is_bce=0)

        iteration = 20
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=10.0, edge_weight=10.0, lpip_weight=1.0, is_bce=0)

        iteration = 21
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=10.0, edge_weight=0.0, lpip_weight=1.0, is_bce=0)

        iteration = 22
        self.iteration_table[iteration] = IterationParameters(iteration, l1_weight=0.0, edge_weight=10.0, lpip_weight=1.0, is_bce=0)

    def get_version(self, iteration):
        return self.iteration_table[iteration].get_version()

    def get_l1_weight(self, iteration):
        return self.iteration_table[iteration].get_l1_weight()

    def get_lpip_weight(self, iteration):
        return self.iteration_table[iteration].get_lpip_weight()

    def get_edge_weight(self, iteration):
        return self.iteration_table[iteration].get_edge_weight()

    def is_bce_enabled(self, iteration):
        return self.iteration_table[iteration].is_bce_enabled()


class IterationParameters():
    def __init__(self, iteration, l1_weight, lpip_weight, edge_weight, is_bce):
        self.iteration = iteration
        self.l1_weight = l1_weight
        self.lpip_weight = lpip_weight
        self.edge_weight = edge_weight
        self.is_bce = is_bce

    def get_version(self):
        return self.iteration

    def get_l1_weight(self):
        return self.l1_weight

    def get_lpip_weight(self):
        return self.lpip_weight

    def get_edge_weight(self):
        return self.edge_weight

    def is_bce_enabled(self):
        return self.is_bce