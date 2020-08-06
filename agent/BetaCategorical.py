import torch
from torch.distributions import Beta, Categorical
import numpy as np
from rlpyt.distributions.base import Distribution
from rlpyt.distributions.discrete import DiscreteMixin
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import valid_mean
from agent.const import IDX_AGENT_ACC, IDX_AGENT_EXP, IDX_AGENT_REJ
from constants import EPS

DistInfo = namedarraytuple("DistInfo", ["pi", "a", "b"])


class BetaCategorical(DiscreteMixin, Distribution):

    def kl(self, old_dist_info, new_dist_info):
        raise NotImplementedError

    def mean_kl(self, old_dist_info, new_dist_info, valid):
        raise NotImplementedError

    def entropy(self, dist_info):
        pi, a, b = self._unpack(dist_info)
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)

        # approximate probability of each concession using midpoint
        dim = torch.from_numpy(np.arange(1, 100) / 100).float().unsqueeze(0)
        x = dim.expand(a.size()[0], -1)
        p = .01 * torch.exp(Beta(a, b).log_prob(x))
        p[:, [0, -1]] *= 1.5
        p /= p.sum(-1, True)

        # multiply by P(concession)
        p *= pi[:, [-1]]

        # add accept and reject (and expire) probabilities
        k = pi.size()[-1]
        assert k in [3, 4]
        if k == 3:
            p = torch.cat([pi[:, [0]], p, pi[:, [1]]], dim=1)
        else:
            p = torch.cat([pi[:, [0]], p, pi[:, [1, 2]]], dim=1)
        assert torch.max(torch.abs(p.sum(-1) - 1.)) < 1e-6

        # calculate entropy of categorical distribution
        entropy = -torch.sum(p * torch.log(p + EPS), dim=-1)

        return entropy

    def perplexity(self, dist_info):
        return torch.exp(self.entropy(dist_info))

    def mean_entropy(self, dist_info, valid=None):
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid=None):
        return valid_mean(self.perplexity(dist_info), valid)

    def log_likelihood(self, x, dist_info):
        pi, a, b = self._unpack(dist_info)
        k = pi.size()[-1]
        assert k in [3, 4]
        lnl = torch.zeros_like(a, device=x.device)

        # rejects
        rej = x == 0
        if torch.sum(rej) > 0:
            lnl[rej] = torch.log(pi[rej, IDX_AGENT_REJ] + EPS)

        # accepts
        acc = x == 100
        if torch.sum(acc) > 0:
            lnl[acc] = torch.log(pi[acc, IDX_AGENT_ACC] + EPS)

        # seller expirations
        if k == 4:
            exp = x == 101
            if torch.sum(exp) > 0:
                lnl[exp] = torch.log(pi[exp, IDX_AGENT_EXP] + EPS)

        # concessions
        con = (x < 100) & (x > 0)
        if torch.sum(con) > 0:
            x_con = x[con].float() / 100
            a_con = a[con]
            b_con = b[con]
            lnl[con] = torch.log(pi[con, k - 1] + EPS) \
                + Beta(a_con, b_con).log_prob(x_con)

        return lnl

    def likelihood_ratio(self, x, old_dist_info, new_dist_info):
        logli_old = self.log_likelihood(x, old_dist_info)
        logli_new = self.log_likelihood(x, new_dist_info)
        return torch.exp(logli_new - logli_old)

    def sample_loglikelihood(self, dist_info):
        raise NotImplementedError

    def sample(self, dist_info):
        pi, a, b = self._unpack(dist_info)
        k = pi.size()[-1]
        assert k in [3, 4]

        # first, draw from categorical
        samples = Categorical(probs=pi).sample()

        # accepts
        acc = samples == IDX_AGENT_ACC
        if torch.sum(acc) > 0:
            samples[acc] = 100

        # seller expirations
        if k == 4:
            exp = samples == IDX_AGENT_EXP
            if torch.sum(exp) > 0:
                samples[exp] = 101

        # concessions
        con = samples == k-1
        if torch.sum(con) > 0:
            a_con, b_con = a[con], b[con]
            p = Beta(a_con, b_con).sample()
            p *= 100
            samples[con] = torch.clamp(torch.round(p).long(), 1, 99)

        return samples

    @staticmethod
    def _unpack(dist_info):
        return dist_info.pi, dist_info.a, dist_info.b
