import torch
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from torch.nn.functional import softplus, softmax
import numpy as np
from scipy.special import betainc
from nets.FeedForward import FeedForward
from utils import load_sizes
from agent.const import DROPOUT_POLICY, DROPOUT_VALUE
from constants import POLICY_SLR, POLICY_BYR

TEST_BETA = False


class AgentModel(torch.nn.Module):
    """
    Agent for eBay simulation.
    1. Fully separate networks for value and policy networks
    2. Policy network outputs parameters of sampling distribution
    3. Value network outputs a scalar between 0 and 1
    4. Both networks use batch normalization
    5. Both networks use dropout with shared dropout hyperparameters
    """
    def __init__(self, byr=None, serial=False, nocon=False):
        """
        Initializes feed-forward networks for agent.
        :param bool byr: use buyer sizes if True.
        :param bool serial: serial sampler doesn't like squeezed outputs.
        :param bool nocon: accept or reject only if True.
        """
        super().__init__()

        # save params to self
        self.byr = byr
        self.serial = serial
        self.out = 1 if nocon else 5

        # policy net
        sizes = load_sizes(POLICY_BYR if byr else POLICY_SLR)
        sizes['out'] = self.out
        self.policy_net = FeedForward(sizes=sizes, dropout=DROPOUT_POLICY)

        # value net
        sizes['out'] = 1
        self.value_net = FeedForward(sizes=sizes, dropout=DROPOUT_VALUE)

        # for discrete pdf
        self.dim = torch.from_numpy(np.linspace(0, 1, 100)).float()

    def forward(self, observation, prev_action=None, prev_reward=None):
        """
        Predicts policy distribution and state value.
        :param namedtuplearray observation: contains dict of agent inputs
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
        if self.out == 1:
            pdf = self._cat_from_binary(theta)
        else:
            pdf = self._cat_from_beta(theta)

        # value
        v = torch.sigmoid(self.value_net(input_dict))

        # squeeze
        if not self.serial:
            pdf = pdf.squeeze()
            v = v.squeeze()

        return pdf, v

    @staticmethod
    def _cat_from_binary(theta):
        p_accept = torch.sigmoid(theta)
        p_reject = 1. - p_accept
        zeros = torch.zeros_like(p_accept).expand(-1, 99)
        pdf = torch.cat([p_reject, zeros, p_accept], dim=-1)
        return pdf

    def _cat_from_beta(self, theta):
        # beta distribution
        pi = softmax(theta[:, :-2], dim=-1)
        beta_params = softplus(torch.clamp(theta[:, -2:], min=-5))
        a, b = beta_params[:, [0]], beta_params[:, [1]]

        # get cdf of concessions, convert to pdf
        x = self.dim.expand(a.size()[0], -1).to(a.device)
        cdf = self._beta_cdf(a, b, x)
        pdf = cdf[:, 1:] - cdf[:, :-1]
        assert torch.max(torch.abs(pdf.sum(-1) - 1.)) < 1e-6

        if TEST_BETA:
            test = betainc(a.cpu().detach().numpy(),
                           b.cpu().detach().numpy(),
                           x.cpu().detach().numpy())
            check = np.max(np.abs(cdf.cpu().detach().numpy() - test))
            if np.any(np.isnan(check)) or check > 1e-6:
                print(torch.cat([a, b], dim=-1))
                print(check)
                print(np.isnan(test).sum())
                print(torch.isnan(cdf).sum().item())
                raise ValueError('Distributions are not equivalent.')

        # add in accepts and rejects
        pdf *= pi[:, [-1]]  # scale by concession probability
        pdf = torch.cat([pi[:, [0]], pdf, pi[:, [1]]], dim=-1)
        assert torch.min(pdf) >= 0
        total = pdf.sum(-1, True)
        if torch.max(torch.abs(total - 1.)) >= 1e-6:
            print('Warning: total probability is {}.'.format(total))
        pdf /= total  # for precision

        return pdf

    @staticmethod
    def _beta_cdf(a, b, x, n_iter=41):
        beta = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))

        # split values
        S0 = (a + 1) / (a + b + 2)
        x_l = torch.min(x, S0)
        x_h = torch.max(x, S0)

        # low values
        T = torch.zeros_like(x)
        for k in reversed(range(n_iter)):
            v = -(a + k) * (a + b + k) * x_l / (a + 2 * k) / (a + 2 * k + 1)
            T = v / (T + 1)
            if k > 0:
                v = k * (b - k) * x_l / (a + 2 * k - 1) / (a + 2 * k)
                T = v / (T + 1)

        cdf_l = x_l ** a * (1 - x_l) ** b / (a * beta) / (T + 1)

        # high values
        T = torch.zeros_like(x)
        for k in reversed(range(n_iter)):
            v = -(b + k) * (a + b + k) * (1 - x_h) / (b + 2 * k) / (b + 2 * k + 1)
            T = v / (T + 1)
            if k > 0:
                v = k * (a - k) * (1 - x_h) / (b + 2 * k - 1) / (b + 2 * k)
                T = v / (T + 1)

        cdf_h = 1 - x_h ** a * (1 - x_h) ** b / (b * beta) / (T + 1)

        # concatenate
        cdf = cdf_l * (x <= S0) + cdf_h * (x > S0)
        return torch.clamp(cdf, min=0., max=1.)


class SplitCategoricalPgAgent(CategoricalPgAgent):
    def value_parameters(self):
        return self.model.value_net.parameters()

    def policy_parameters(self):
        return self.model.policy_net.parameters()