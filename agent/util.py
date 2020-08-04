import os
import numpy as np
from torch.distributions import Beta
from rlenv.util import sample_categorical, proper_squeeze
from agent.models.BetaCategorical import DistInfo
from agent.const import IDX_AGENT_REJ, IDX_AGENT_ACC, IDX_AGENT_EXP
from constants import AGENT_DIR, BYR, SLR, POLICY_SLR, POLICY_BYR


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    return log_dir


def get_run_id(delta=None, entropy_coeff=None, use_kl=False, suffix=None):
    start = 'kl' if use_kl else 'entropy'
    end = '{}_delta_{}'.format(entropy_coeff, delta)
    if suffix is not None:
        end += '_{}'.format(suffix)
    return '{}_{}'.format(start, end)


def sample_beta_categorical(pi=None, a=None, b=None):
    k = pi.size()[-1]
    assert k in [3, 4]
    action = int(sample_categorical(probs=pi))
    if action == IDX_AGENT_REJ:  # reject
        return 0
    if action == IDX_AGENT_ACC:  # accept
        return 100
    if k == 4 and action == IDX_AGENT_EXP:  # seller expiration
        return 101
    if action == k-1:  # concession
        beta = Beta(a, b)
        sample = proper_squeeze(beta.sample(sample_shape=(1,)).float())
        con = int(np.round(sample.numpy() * 100))
        if con == 0:  # cannot reject when drawing a concession
            con = 1
        if con == 100:  # cannot accept when drawing a concession
            con = 99
        return con


def pack_dist_info(params=None):
    pi, a, b = params
    return DistInfo(pi=pi, a=a, b=b)