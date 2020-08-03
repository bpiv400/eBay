import os
from constants import AGENT_DIR, BYR, SLR, POLICY_SLR, POLICY_BYR


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    return log_dir


def get_run_id(delta=None, entropy_coeff=None, use_kl=False):
    prefix = 'kl' if use_kl else 'entropy'
    suffix = '{}_delta_{}'.format(entropy_coeff, delta)
    return '{}_{}'.format(prefix, suffix)
