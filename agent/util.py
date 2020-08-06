import os
from agent.BetaCategorical import DistInfo
from constants import AGENT_DIR, POLICY_SLR, POLICY_BYR
from featnames import SLR, BYR


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_paths(byr=None, delta=None, entropy_coeff=None, suffix=None):
    # log directory
    if byr:
        log_dir = AGENT_DIR + '{}/'.format(BYR)
    else:
        log_dir = AGENT_DIR + '{}/delta_{}/'.format(SLR, delta)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # run id
    run_id = 'entropy_{}'.format(entropy_coeff)
    if suffix is not None:
        run_id += '_{}'.format(suffix)

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir


def pack_dist_info(params=None):
    pi, a, b = params
    return DistInfo(pi=pi, a=a, b=b)
