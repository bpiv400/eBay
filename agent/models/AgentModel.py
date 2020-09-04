import torch
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from torch.nn.functional import softmax
import numpy as np
from nets.FeedForward import FeedForward
from agent.util import define_con_set
from utils import load_sizes
from constants import POLICY_SLR, POLICY_BYR


class AgentModel(torch.nn.Module):
    """
    Agent for eBay simulation.
    1. Fully separate networks for value and policy networks
    2. Policy network outputs parameters of sampling distribution
    3. Value network outputs a scalar between 0 and 1
    4. Both networks use batch normalization
    5. Both networks use dropout with shared dropout hyperparameters
    """
    def __init__(self, byr=None, dropout=None, serial=False, con_set=False):
        """
        Initializes feed-forward networks for agents.
        :param bool byr: use buyer sizes if True.
        :param tuple dropout: pair of dropout rates.
        :param bool serial: serial sampler doesn't like squeezed outputs.
        :param str con_set: restricts concession set.
        """
        super().__init__()

        # save params to self
        self.byr = byr
        self.serial = serial
        self.con_set = con_set

        # size of policy output
        self.out = len(define_con_set(byr=byr, con_set=con_set))

        # policy net
        sizes = load_sizes(POLICY_BYR if byr else POLICY_SLR)
        sizes['out'] = self.out
        self.policy_net = FeedForward(sizes=sizes, dropout=dropout)

        # value net
        sizes['out'] = 1
        self.value_net = FeedForward(sizes=sizes, dropout=dropout)

        # for discrete pdf
        self.dim = torch.from_numpy(np.linspace(0, 1, 100)).float()

    def forward(self, observation, prev_action=None, prev_reward=None):
        """
        Predicts policy distribution and state value.
        :param namedtuplearray observation: contains dict of agents inputs
        :param None prev_action: (not used; for recurrent agents only)
        :param None prev_reward: (not used; for recurrent agents only)
        :return: tuple of policy distribution, value
        """
        # noinspection PyProtectedMember
        input_dict = observation._asdict()

        # produce warning if 0-tensor is passed in observation
        x_lstg = input_dict['lstg']
        if len(x_lstg.size()) == 1 and x_lstg.sum() == 0.:
            print('Warning: should only occur in initialization')

        # processing for single observations
        if input_dict['lstg'].dim() == 1:
            for elem_name, elem in input_dict.items():
                input_dict[elem_name] = elem.unsqueeze(0)
            if self.training:
                self.eval()

        # policy
        theta = self.policy_net(input_dict)

        # convert to categorical distribution
        pdf = softmax(theta, dim=-1)

        # value
        v = torch.sigmoid(self.value_net(input_dict))

        # squeeze
        if not self.serial:
            pdf = pdf.squeeze()
            v = v.squeeze()

        return pdf, v


class SplitCategoricalPgAgent(CategoricalPgAgent):
    def value_parameters(self):
        return self.model.value_net.parameters()

    def policy_parameters(self):
        return self.model.policy_net.parameters()