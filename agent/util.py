import os
from rlenv.environments.SellerEnv import RelistSellerEnv, NoRelistSellerEnv
from rlenv.environments.BuyerEnv import BuyerEnv
from constants import AGENT_DIR, POLICY_SLR, POLICY_BYR
from featnames import SLR, BYR


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_paths(**kwargs):
    # log directory
    if kwargs[BYR]:
        log_dir = AGENT_DIR + '{}/'.format(BYR)
    else:
        log_dir = AGENT_DIR + '{}/delta_{}/'.format(SLR, kwargs['delta'])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # run id
    run_id = 'entropy_{}'.format(kwargs['entropy_coeff'])
    if 'suffix' in kwargs:
        run_id += '_{}'.format(kwargs['suffix'])

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir


def get_env(byr=None, delta=None):
    if byr:
        return BuyerEnv
    if delta == 0.:
        return NoRelistSellerEnv
    return RelistSellerEnv
