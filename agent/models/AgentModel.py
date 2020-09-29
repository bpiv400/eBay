import numpy as np
import torch
from torch.nn.functional import softmax
from nets.FeedForward import FeedForward
from agent.util import define_con_space
from utils import load_sizes
from agent.const import FULL, AGENT_HIDDEN, BYR_MIN_CON1, AGENT_STATE
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
    def __init__(self, byr=None, con_set=None, dropout=None, value=True):
        """
        Initializes feed-forward networks for agents.
        :param bool byr: use buyer sizes if True
        :param str con_set: restricts concession set
        :param tuple dropout: pair of dropout rates
        :param bool value: estimate value if true
        """
        super().__init__()

        # save params to self
        self.byr = byr
        self.con_set = con_set
        self.value = value

        # size of policy output
        self.out = len(define_con_space(byr=byr, con_set=con_set))

        # policy net
        sizes = load_sizes(POLICY_BYR if byr else POLICY_SLR)
        sizes['out'] = self.out
        self.policy_net = FeedForward(sizes=sizes,
                                      hidden=AGENT_HIDDEN,
                                      dropout=dropout)

        # value net
        if self.value:
            sizes['out'] = 6 if byr else 5
            self.value_net = FeedForward(sizes=sizes,
                                         hidden=AGENT_HIDDEN,
                                         dropout=dropout)

    def forward(self, observation, prev_action=None, prev_reward=None, value_only=False):
        """
        Predicts policy distribution and state value.
        :param namedtuplearray observation: contains dict of agents inputs
        :param None prev_action: (not used; for recurrent agents only)
        :param None prev_reward: (not used; for recurrent agents only)
        :param bool value_only: only return value if True
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

        if self.value:
            value_params = self.value_net(input_dict)
            if value_only:
                return value_params
            else:
                pdf = self._get_pdf(input_dict)
                return pdf, value_params
        else:
            return self._get_pdf(input_dict)

    def _get_pdf(self, input_dict):
        theta = self.policy_net(input_dict)
        if self.byr:
            # no small concessions on turn 1
            t1 = input_dict[LSTG][:, -3] == 1
            if self.con_set == FULL:
                theta[t1, 1:BYR_MIN_CON1] = -np.inf
            else:
                upper = int(BYR_MIN_CON1 / 10)
                theta[t1, 1:upper] = -np.inf

            # accept or reject on turn 7
            t7 = torch.sum(input_dict[LSTG][:, [-3, -2, -1]], dim=1) == 0
            theta[t7, 0] = 0.
            if self.con_set == FULL:
                theta[t7, 1:100] = -np.inf
            else:
                theta[t7, 1:10] = -np.inf

        return softmax(theta, dim=-1)


def load_agent_model(model_args=None, run_dir=None):
    model = AgentModel(**model_args)
    path = run_dir + 'params.pkl'
    d = torch.load(path, map_location=torch.device('cpu'))
    if AGENT_STATE in d:
        d = d[AGENT_STATE]
    d = {k: v for k, v in d.items() if not k.startswith('value')}
    model.load_state_dict(d, strict=True)
    for param in model.parameters(recurse=True):
        param.requires_grad = False
    model.eval()
    return model
