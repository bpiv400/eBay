import torch
from torch.nn.functional import softmax
from nets.FeedForward import FeedForward
from utils import load_sizes
from agent.const import AGENT_STATE
from constants import SLR, BYR, NUM_COMMON_CONS
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
    def __init__(self, byr=None, dropout=None, value=True):
        """
        Initializes feed-forward networks for agents.
        :param bool byr: use buyer sizes if True
        :param tuple dropout: pair of dropout rates
        :param bool value: estimate value if true
        """
        super().__init__()

        # save params to self
        self.value = value

        # save x sizes for coverting tensor input to dictionary
        sizes = load_sizes(BYR if byr else SLR)
        self.x_sizes = sizes['x']

        # in case of no dropout
        if dropout is None:
            dropout = (0., 0.)

        # policy net
        sizes['out'] = NUM_COMMON_CONS + (2 if byr else 3)
        self.policy_net = FeedForward(sizes=sizes, dropout=dropout)

        # value net
        if self.value:
            sizes['out'] = 6 if byr else 5
            self.value_net = FeedForward(sizes=sizes, dropout=dropout)

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
        x = observation._asdict()

        # processing for single observations
        x_lstg = x[LSTG]
        if x_lstg.dim() == 1:
            if x_lstg.sum == 0:
                print('Warning: should only occur in initialization')
            for elem_name, elem in x.items():
                x[elem_name] = elem.unsqueeze(0)
            if self.training:
                self.eval()

        if self.value:
            value_params = self.value_net(x)
            if value_only:
                return value_params
            else:
                pdf = self._get_pdf(x)
                return pdf, value_params
        else:
            return self._get_pdf(x)

    def _get_pdf(self, x):
        theta = self.policy_net(x)

        # for shapley values, get x as dict
        if type(theta) is tuple:
            theta, x = theta

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
