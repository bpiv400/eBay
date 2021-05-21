import numpy as np
import torch
from torch.distributions import Beta
from agent.agents.SplitCategoricalPgAgent import SplitCategoricalPgAgent
from agent.agents.util import parse_value_params
from constants import EPS, IDX
from featnames import SLR


class SellerAgent(SplitCategoricalPgAgent):

    def _calculate_value(self, value_params):
        p, a, b = parse_value_params(value_params)
        beta_mean = a / (a + b)
        v = p[:, 1] + p[:, -1] * beta_mean
        return v

    def get_value_loss(self, value_params, return_, valid):
        p, a, b = parse_value_params(value_params)
        zeros = torch.zeros_like(return_)

        # no sale and worth zero
        idx0 = torch.isclose(return_, zeros) & valid
        lnL = torch.sum(torch.log(p[idx0, 0] + EPS))

        # sells for list price
        idx1 = torch.isclose(return_, zeros + 1) & valid
        lnL += torch.sum(torch.log(p[idx1, 1] + EPS))

        # intermediate outcome
        idx_beta = ~idx0 & ~idx1 & valid
        lnL += torch.sum(torch.log(p[idx_beta, -1] + EPS))
        dist = Beta(a[idx_beta], b[idx_beta])
        lnL += torch.sum(dist.log_prob(return_[idx_beta]))

        return -lnL

    def _get_turn_info(self, opt_info=None, env=None, turn=None,
                       con=None, valid=None):
        for t in IDX[SLR]:
            # rate of reaching turn t
            opt_info['Rate_{}'.format(t)] = np.mean(turn == t)

            # accept, reject, and expiration rates
            con_t = con[turn == t]
            prefix = 'Turn{}'.format(t)
            opt_info['{}_{}'.format(prefix, 'AccRate')] = np.mean(con_t == 1)
            opt_info['{}_{}'.format(prefix, 'ExpRate')] = np.mean(con_t > 1)
            opt_info['{}_{}'.format(prefix, 'RejRate')] = np.mean(con_t == 0)

            # moments of concession distribution
            opt_info['{}_{}'.format(prefix, 'ConRate')] = \
                np.mean((con_t > 0) & (con_t < 1))
            opt_info['{}{}'.format(prefix, 'Con')] = \
                con_t[(con_t > 0) & (con_t < 1)]
        return opt_info
