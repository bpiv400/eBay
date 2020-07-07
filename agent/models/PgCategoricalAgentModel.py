import torch
from torch.nn.functional import softmax
from utils import load_sizes
from agent.models.AgentModel import AgentModel
from nets.FeedForward import FeedForward
from constants import POLICY_SLR, POLICY_BYR


class PgCategoricalAgentModel(AgentModel):
    """
    Agent for eBay simulation

    1. Fully separate networks for value and policy networks
    2. Policy network outputs categorical probability distribution
     over actions
    3. Both networks use batch normalization
    4. Both networks use dropout and share dropout hyperparameters
    """
    def __init__(self, **kwargs):
        """
        kwargs:
            sizes: gives all sizes for the model, including size
            of each input grouping 'x' and number of elements in
            output vector 'out'
        """
        super().__init__(**kwargs)

        # policy net
        sizes = load_sizes(POLICY_BYR if self.byr else POLICY_SLR)
        self.policy_network = FeedForward(sizes=sizes,
                                          dropout=self.dropout_policy)

        # value net
        sizes['out'] = 1
        self.value_network = FeedForward(sizes=sizes,
                                         dropout=self.dropout_value)

    def value_parameters(self):
        return self.value_network.parameters()

    def policy_parameters(self):
        return self.policy_network.parameters()

    def con(self, input_dict=None):
        logits, _ = self._forward_dict(input_dict=input_dict,
                                       compute_value=False)
        logits = logits.squeeze()
        return logits

    def pi(self, observation, prev_action, prev_reward):
        input_dict = observation._asdict()
        logits = self.con(input_dict=input_dict)
        pi = softmax(logits, dim=logits.dim() - 1)
        return pi

    def _forward_dict(self, input_dict=None, compute_value=True):
        if input_dict['lstg'].dim() == 1:
            for elem_name, elem in input_dict.items():
                input_dict[elem_name] = elem.unsqueeze(0)
            if self.training:
                self.eval()
        pi_logits = self.policy_network(input_dict)
        # value output head
        if compute_value:
            v = torch.sigmoid(self.value_network(input_dict))
        else:
            v = None

        return pi_logits, v

    def forward(self, observation, prev_action, prev_reward):
        """
        :return: tuple of pi, v
        """
        input_dict = observation._asdict()
        logits, v = self._forward_dict(input_dict=input_dict,
                                       compute_value=True)
        pi = softmax(logits, dim=logits.dim() - 1)

        # transformations
        pi = pi.squeeze()
        v = v.squeeze()

        return pi, v
