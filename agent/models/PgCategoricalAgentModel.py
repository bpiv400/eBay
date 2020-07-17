import torch
from torch.nn.functional import softmax
from utils import load_sizes
from nets.FeedForward import FeedForward
from agent.const import T1_IDX, DELAY_BOOST
from constants import POLICY_SLR, POLICY_BYR
from rlenv.util import sample_categorical


class PgCategoricalAgentModel(torch.nn.Module):
    """
    Agent for eBay simulation

    1. Fully separate networks for value and policy networks
    2. Policy network outputs categorical probability distribution
     over actions
    3. Value network outputs a scalar between 0 and 1
    4. Both networks use batch normalization
    5. Both networks use dropout with separate dropout hyperparameters
    """
    def __init__(self, byr=None, dropout=None):
        super().__init__()
        self.byr = byr

        # policy net
        sizes = load_sizes(POLICY_BYR if self.byr else POLICY_SLR)
        self.policy_net = FeedForward(sizes=sizes, dropout=dropout)

        # value net
        sizes['out'] = 1
        self.value_net = FeedForward(sizes=sizes, dropout=dropout)

    def value_parameters(self):
        return self.value_net.parameters()

    def policy_parameters(self):
        return self.policy_net.parameters()

    def con(self, input_dict=None):
        logits, _ = self._forward_dict(input_dict=input_dict,
                                       compute_value=False)
        logits = logits.squeeze()
        return sample_categorical(logits=logits)

    def _forward_dict(self, input_dict=None, compute_value=True):
        # processing for single observations
        if input_dict['lstg'].dim() == 1:
            for elem_name, elem in input_dict.items():
                input_dict[elem_name] = elem.unsqueeze(0)
            if self.training:
                self.eval()

        # policy
        pi_logits = self.policy_net(input_dict)

        # bias towards waiting for buyer's first turn
        if self.byr:
            first_turn = input_dict['lstg'][:, T1_IDX] == 1
            pi_logits[first_turn, 0] += DELAY_BOOST

        # value
        if compute_value:
            v = torch.sigmoid(self.value_net(input_dict))
        else:
            v = None
        return pi_logits, v

    def forward(self, observation, prev_action, prev_reward):
        """
        :return: tuple of pi, v
        """
        # noinspection PyProtectedMember
        input_dict = observation._asdict()

        # get policy and value
        pi_logits, v = self._forward_dict(input_dict=input_dict,
                                          compute_value=True)

        pi = softmax(pi_logits, dim=pi_logits.dim() - 1)

        # transformations
        pi = pi.squeeze()
        v = v.squeeze()

        return pi, v
