import torch
from torch.distributions import Beta, Categorical
from rlpyt.distributions.base import Distribution
from rlpyt.distributions.discrete import DiscreteMixin
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import valid_mean
from agent.const import IDX_AGENT_REJ, IDX_AGENT_ACC, IDX_AGENT_EXP
from constants import EPS

DistInfo = namedarraytuple("DistInfo", ["pi", "a", "b"])


class BetaCategorical(DiscreteMixin, Distribution):

    def kl(self, old_dist_info, new_dist_info):
        raise NotImplementedError

    def mean_kl(self, old_dist_info, new_dist_info, valid):
        raise NotImplementedError

    def entropy(self, dist_info):
        pi, a, b = self._unpack(dist_info)
        cat_entropy = Categorical(probs=pi).entropy()
        beta_entropy = Beta(a, b).entropy()
        return cat_entropy + beta_entropy

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
            lnl[con] = torch.log(pi[con, k - 1] + EPS)
            lnl[con] += - self._log_beta_function(a_con, b_con) \
                + (a_con-1) * torch.log(x_con) \
                + (b_con-1) * torch.log(1-x_con)

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

    @staticmethod
    def _log_beta_function(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
