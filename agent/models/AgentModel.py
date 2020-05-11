import torch.nn as nn


class AgentModel(nn.Module):
    def __init__(self, delay=False, byr=False):
        super().__init__()
        self.delay = delay
        self.byr = byr

    def con(self, input_dict=None):
        raise NotImplementedError()

    def forward(self, observation, prev_action, prev_reward):
        raise NotImplementedError()
