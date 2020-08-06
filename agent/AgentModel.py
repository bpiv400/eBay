import torch
from torch.nn.functional import softmax, softplus
from nets.FeedForward import FeedForward
from utils import load_sizes
from agent.const import T1_IDX, DELAY_BOOST, DROPOUT_POLICY, DROPOUT_VALUE, \
    IDX_AGENT_REJ, NUM_ACTIONS_SLR, NUM_ACTIONS_BYR, NUM_PARAM_SLR, \
    NUM_PARAM_BYR
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
    def __init__(self, byr=None):
        super().__init__()
        self.byr = byr

        # policy net
        sizes = load_sizes(POLICY_BYR if self.byr else POLICY_SLR)
        sizes['out'] = NUM_PARAM_BYR if self.byr else NUM_PARAM_SLR
        self.policy_net = FeedForward(sizes=sizes, dropout=DROPOUT_POLICY)
        self.num_out = sizes['out']
        self.x_sizes = sizes['x']

        # value net
        sizes['out'] = 1
        self.value_net = FeedForward(sizes=sizes, dropout=DROPOUT_VALUE)

    def forward(self, observation, prev_action=None, prev_reward=None):
        """
        Predicts policy parameters and state value
        :param namedtuplearray observation: contains dict of agent inputs
        :param None prev_action: (not used; for recurrent agents only)
        :param None prev_reward: (not used; for recurrent agents only)
        :return: tuple of params, v
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

        # bias towards waiting for buyer's first turn
        if self.byr:
            first_turn = input_dict['lstg'][:, T1_IDX] == 1
            theta[first_turn, IDX_AGENT_REJ] += DELAY_BOOST

        # beta-categorical distribution
        if self.num_out in [5, 6]:
            pi = softmax(theta[:, :-2], dim=-1).squeeze()
            beta_params = softplus(torch.clamp(theta[:, -2:], min=-5)) + 1  # 0 mass at {0,1}
            a = beta_params[:, 0].squeeze()
            b = beta_params[:, 1].squeeze()
            params = (pi, a, b)

        # categorical distribution
        elif self.num_out in [NUM_ACTIONS_SLR, NUM_ACTIONS_BYR]:
            params = softmax(theta, dim=theta.dim() - 1).squeeze()

        else:
            raise RuntimeError("Incorrect sizes['out']")

        # value
        v = torch.sigmoid(self.value_net(input_dict)).squeeze()

        return params, v
