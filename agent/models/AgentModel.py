import numpy as np
import torch
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.distributions.categorical import DistInfo
from rlpyt.utils.buffer import buffer_to
from torch.nn.functional import softmax, softplus
from nets.FeedForward import FeedForward
from agent.util import define_con_set
from utils import load_sizes
from agent.const import DROPOUT, FULL
from constants import POLICY_SLR, POLICY_BYR
from featnames import LSTG


class AgentModel(torch.nn.Module):
    """
    Agent for eBay simulation.
    1. Fully separate networks for value and policy networks
    2. Policy network outputs parameters of sampling distribution
    3. Value network outputs a scalar between 0 and 1
    4. Both networks use batch normalization
    5. Both networks use dropout with shared dropout hyperparameters
    """
    def __init__(self, byr=None, serial=False, con_set=None, value=True):
        """
        Initializes feed-forward networks for agents.
        :param bool byr: use buyer sizes if True.
        :param bool serial: serial sampler doesn't like squeezed outputs.
        :param str con_set: restricts concession set.
        :param bool value: estimate value if true
        """
        super().__init__()

        # save params to self
        self.byr = byr
        self.serial = serial
        self.con_set = con_set
        self.value = value

        # size of policy output
        self.out = len(define_con_set(byr=byr, con_set=con_set))

        # policy net
        sizes = load_sizes(POLICY_BYR if byr else POLICY_SLR)
        sizes['out'] = self.out
        self.policy_net = FeedForward(sizes=sizes, dropout=DROPOUT)

        # value net
        if self.value:
            sizes['out'] = 4 if byr else 5
            self.value_net = FeedForward(sizes=sizes, dropout=DROPOUT)

    def forward(self, observation, prev_action=None, prev_reward=None, vparams=False):
        """
        Predicts policy distribution and state value.
        :param namedtuplearray observation: contains dict of agents inputs
        :param None prev_action: (not used; for recurrent agents only)
        :param None prev_reward: (not used; for recurrent agents only)
        :param bool vparams: return parameters of value distribution if True, mean if False
        :return: tuple of policy distribution, value
        """
        # noinspection PyProtectedMember
        input_dict = observation._asdict()

        # processing for single observations
        x_lstg = input_dict[LSTG]
        if x_lstg.dim() == 1:
            if x_lstg.sum == 0:
                print('Warning: should only occur in initialization')
            for elem_name, elem in input_dict.items():
                input_dict[elem_name] = elem.unsqueeze(0)
            if self.training:
                self.eval()

        # policy as categorical distribution
        theta = self.policy_net(input_dict)
        if self.byr:
            # no small concessions on turn 1
            t1 = input_dict[LSTG][:, -3] == 1
            if self.con_set == FULL:
                theta[t1, 1:40] = -np.inf
            else:
                theta[t1, 1:4] = -np.inf

            # accept or reject on turn 7
            t7 = torch.sum(input_dict[LSTG][:, [-3, -2, -1]], dim=1) == 0
            theta[t7, 1:100] = -np.inf

        pdf = softmax(theta, dim=-1)
        if not self.serial:
            pdf = pdf.squeeze()

        if not self.value:
            return pdf

        # value
        value_params = self.value_net(input_dict)
        pi = softmax(value_params[:, :-2], dim=-1)
        beta_params = softplus(torch.clamp(value_params[:, -2:], min=-5))
        a, b = beta_params[:, 0], beta_params[:, 1]

        if vparams:
            if not self.serial:
                pi = pi.squeeze()
                a = a.squeeze()
                b = b.squeeze()
            return pdf, (pi, a, b)

        if self.byr:  # values range from -1 to 1
            v = pi[:, -1] * (a / (a + b) * 2 - 1)
        else:
            v += pi[:, 1] + pi[:, -1] * a / (a + b)

        if not self.serial:
            v = v.squeeze()

        return pdf, v


class SplitCategoricalPgAgent(CategoricalPgAgent):

    def __call__(self, observation, prev_action, prev_reward, vparams=False):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        pi, v = self.model(*model_inputs, vparams=vparams)
        return buffer_to((DistInfo(prob=pi), v), device="cpu")

    def value_parameters(self):
        return self.model.value_net.parameters()

    def policy_parameters(self):
        return self.model.policy_net.parameters()
