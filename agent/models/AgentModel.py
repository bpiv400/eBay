import torch.nn as nn


class AgentModel(nn.Module):
    def __init__(self,
                 byr=False,
                 dropout_policy=None,
                 dropout_value=None):
        super().__init__()
        self.byr = byr
        self.dropout_policy = dropout_policy
        self.dropout_value = dropout_value

    def con(self, input_dict=None):
        raise NotImplementedError()

    def forward(self, observation, prev_action, prev_reward):
        raise NotImplementedError()
