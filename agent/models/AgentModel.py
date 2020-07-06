import torch.nn as nn


class AgentModel(nn.Module):
    def __init__(self,
                 byr=False,
                 delay=False,
                 dropout_policy=(0., 0.),
                 dropout_value=(0., 0.),
                 pretrained=False):
        super().__init__()
        self.byr = byr
        self.delay = delay
        self.dropout_policy = dropout_policy
        self.dropout_value = dropout_value
        self.pretrained = pretrained

    def con(self, input_dict=None):
        raise NotImplementedError()

    def forward(self, observation, prev_action, prev_reward):
        raise NotImplementedError()
