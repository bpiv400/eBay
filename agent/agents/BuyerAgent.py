import numpy as np
import torch
from torch.distributions import Beta
from agent.agents.SplitCategoricalPgAgent import SplitCategoricalPgAgent
from agent.agents.util import parse_value_params, backward_from_done
from agent.const import AGENT_CONS
from constants import EPS


class BuyerAgent(SplitCategoricalPgAgent):
    def __init__(self, delta=None, turn_cost=0, **kwargs):
        super().__init__(**kwargs)
        self.min = delta - 1  # lowest reward from a sale
        self.max = delta - AGENT_CONS[1][1]  # length of [.5, value]
        self.turn_cost_agent = turn_cost > 0

    def _calculate_value(self, value_params):
        if self.turn_cost_agent:
            return value_params

        p, a, b = parse_value_params(value_params)
        beta_mean = a / (a + b) * (self.max - self.min) + self.min
        v = p[:, 1] * self.min + p[:, 2] * self.max + p[:, -1] * beta_mean
        return v

    def get_value_loss(self, value_params, return_, valid):
        if self.turn_cost_agent:
            value_error = 0.5 * (value_params - return_) ** 2
            return value_error[valid].mean()

        p, a, b = parse_value_params(value_params)
        norm_return = (return_ - self.min) / (self.max - self.min)

        # no sale
        zeros = torch.zeros_like(return_)
        idx0 = torch.isclose(return_, zeros, atol=1e-6) & valid
        lnL = torch.sum(torch.log(p[idx0, 0] + EPS))

        # purchased at list price
        if not np.isclose(self.min, 0):
            idx1 = torch.isclose(norm_return, zeros, atol=1e-6) & valid
            lnL += torch.sum(torch.log(p[idx1, 1] + EPS))
        else:
            idx1 = idx0

        # purchased for half of list price
        idx2 = torch.isclose(norm_return, zeros + 1, atol=1e-6) & valid
        lnL += torch.sum(torch.log(p[idx2, 2] + EPS))

        # intermediate outcome
        idx_beta = ~idx0 & ~idx1 & ~idx2 & valid
        lnL += torch.sum(torch.log(p[idx_beta, -1] + EPS))
        dist = Beta(a[idx_beta], b[idx_beta])
        lnL += torch.sum(dist.log_prob(norm_return[idx_beta]))

        return -lnL

    def _get_turn_info(self, opt_info=None, env=None, turn=None,
                       con=None, valid=None):
        turn_ct = backward_from_done(x=env.env_info.turn, done=env.done)
        turn_ct = turn_ct[valid].numpy()
        for t in [1, 3, 5]:
            # rate of reaching turn t
            opt_info['Rate_{}'.format(t)] = np.mean(turn_ct == t)

            # accept and reject rates and moments of concession distribution
            con_t = con[turn == t]
            prefix = 'Turn{}'.format(t)
            opt_info['{}_{}'.format(prefix, 'AccRate')] = np.mean(con_t == 1)
            opt_info['{}_{}'.format(prefix, 'RejRate')] = np.mean(con_t == 0)
            opt_info['{}_{}'.format(prefix, 'ConRate')] = \
                np.mean((con_t > 0) & (con_t < 1))
            opt_info['{}{}'.format(prefix, 'Con')] = \
                con_t[(con_t > 0) & (con_t < 1)]
        return opt_info
