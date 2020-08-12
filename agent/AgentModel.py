import torch
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
    def __init__(self, byr=None, serial=False):
        super().__init__()

        # save params to self
        self.byr = byr
        self.serial = serial

        # policy net
        sizes = load_sizes(POLICY_BYR if byr else POLICY_SLR)
        sizes['out'] = 5 if byr else 6
        self.policy_net = FeedForward(sizes=sizes, dropout=DROPOUT_POLICY)

        # value net
        sizes['out'] = 1
        self.value_net = FeedForward(sizes=sizes, dropout=DROPOUT_VALUE)

        # for discrete pdf
        self.dim = torch.from_numpy(np.linspace(0, 1, 100)).float()

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

        # beta distribution
        pi = softmax(theta[:, :-2], dim=-1)
        beta_params = softplus(torch.clamp(theta[:, -2:], min=-5))
        a, b = beta_params[:, 0], beta_params[:, 1]

        # convert to categorical distribution
        pdf = self._discrete_pdf(pi, a, b)

        # value
        v = torch.sigmoid(self.value_net(input_dict))

        # squeeze
        if not self.serial:
            pdf = pdf.squeeze()
            v = v.squeeze()

        return pdf, v

    def _discrete_pdf(self, pi, a, b):
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        x = self.dim.expand(a.size()[0], -1).to(a.device)

        # get cdf of concessions, convert to pdf
        cdf = self._beta_cdf(a, b, x)
        assert torch.max(cdf) <= 1
        assert torch.min(cdf) >= 0
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
        if not self.byr:
            pdf = torch.cat([pdf, pi[:, [2]]], dim=-1)  # expiration rejection
        assert torch.min(pdf) >= 0
        assert torch.max(torch.abs(pdf.sum(-1) - 1.)) < 1e-6
        pdf /= pdf.sum(-1, True)  # for precision

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
        return cdf_l * (x <= S0) + cdf_h * (x > S0)
