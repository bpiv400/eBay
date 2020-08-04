import torch
from torch.nn.functional import softmax
from utils import load_sizes
from nets.FeedForward import FeedForward
from agent.const import T1_IDX, DELAY_BOOST, VALUE_DROPOUT, IDX_AGENT_REJ
from constants import POLICY_SLR, POLICY_BYR
from rlenv.util import sample_categorical
from agent.util import sample_beta_categorical


class AgentModel(torch.nn.Module):
    """
    Agent for eBay simulation

    1. Fully separate networks for value and policy networks
    2. Policy network outputs categorical probability distribution
     over actions
    3. Value network outputs a scalar between 0 and 1
    4. Both networks use batch normalization
    5. Both networks use dropout with shared dropout hyperparameters
    """
    def __init__(self, byr=None, dropout=None, model_state_dict=None):
        super().__init__()
        self.byr = byr

        # policy net
        sizes = load_sizes(POLICY_BYR if self.byr else POLICY_SLR)
        self.policy_net = FeedForward(sizes=sizes, dropout=dropout)

        # value net
        sizes['out'] = 1
        self.value_net = FeedForward(sizes=sizes, dropout=VALUE_DROPOUT)

        # initialized model
        if model_state_dict is not None:
            self.load_state_dict(state_dict=model_state_dict,
                                 strict=True)

    def con(self, observation=None):
        params = self._forward_dict(observation=observation,
                                    compute_value=False)
        action = self._sample_action(params)
        return action

    def forward(self, observation, prev_action, prev_reward):
        """
        :return: tuple of params, v
        """
        return self._forward_dict(observation=observation)

    def _forward_dict(self, observation=None, compute_value=True):
        # noinspection PyProtectedMember
        input_dict = observation._asdict()

        # processing for single observations
        if input_dict['lstg'].dim() == 1:
            for elem_name, elem in input_dict.items():
                input_dict[elem_name] = elem.unsqueeze(0)
            if self.training:
                self.eval()

        # policy
        theta = self.policy_net(input_dict)

        # bias towards waiting for buyer's first turn
        if self.byr:
            first_turn = input_dict['lstg'][:, T1_IDX] == 1
            theta[first_turn, IDX_AGENT_REJ] += DELAY_BOOST

        # distribution parameters from model output
        params = self._params_from_output(theta)

        if not compute_value:
            return params

        # value
        v = torch.sigmoid(self.value_net(input_dict)).squeeze()
        return params, v

    @staticmethod
    def _params_from_output(theta):
        raise NotImplementedError()

    @staticmethod
    def _sample_action(params):
        raise NotImplementedError()


class CategoricalAgentModel(AgentModel):
    @staticmethod
    def _params_from_output(theta):
        return softmax(theta, dim=theta.dim() - 1).squeeze()

    @staticmethod
    def _sample_action(params):
        return int(sample_categorical(probs=params))


class BetaCategoricalAgentModel(AgentModel):
    @staticmethod
    def _params_from_output(theta):
        pi = softmax(theta[:, :-2], dim=-1)
        beta_params = torch.clamp(theta[:, -2:], min=1.)  # 0 mass at {0,1}
        a, b = beta_params[:, 0], beta_params[:, 1]
        return pi.squeeze(), a.squeeze(), b.squeeze()

    @staticmethod
    def _sample_action(params):
        pi, a, b = params
        return sample_beta_categorical(pi=pi, a=a, b=b)
