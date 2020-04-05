import torch.nn as nn


class AgentModel(nn.Module):
    def __init__(self, delay=False, byr=False, sizes=None):
        super().__init__()
        self.delay = delay
        self.byr = byr
        self.sizes = sizes

    def con(self, input_dict=None):
        raise NotImplementedError()
