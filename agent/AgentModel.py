import torch
from torch.nn.functional import softmax
from nets.FeedForward import FeedForward
from utils import load_sizes, get_role
from agent.const import NUM_COMMON_CONS
from featnames import LSTG


class AgentModel(torch.nn.Module):
    """
    Agent for eBay simulation.
    1. Fully separate networks for value and policy networks
    2. Policy network outputs parameters of sampling distribution
    3. Value network outputs parameters of value distribution
    4. Both networks use batch normalization
    5. Neither network uses dropout
    """
    def __init__(self, byr=None, value=True, turn_cost=0):
        """
        Initializes feed-forward networks for agents.
        :param bool byr: use buyer sizes if True
        :param bool value: estimate value if true
        :param int turn_cost: for determining size of value network output
        """
        super().__init__()

        # save params to self
        self.value = value
        self.byr = byr

        # save x sizes for coverting tensor input to dictionary
        sizes = load_sizes(get_role(byr))
        self.x_sizes = sizes['x']

        # policy net
        sizes['out'] = NUM_COMMON_CONS + (2 if byr else 3)
        self.policy_net = FeedForward(sizes=sizes)

        # value net
        if self.value:
            sizes['out'] = 1
            self.value_net = FeedForward(sizes=sizes)

    def forward(self, observation, value_only=False):
        """
        Predicts policy distribution and state value.
        :param namedtuplearray observation: contains dict of agents inputs
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
        return softmax(theta, dim=-1)
