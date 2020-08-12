import os
from constants import AGENT_DIR, POLICY_SLR, POLICY_BYR
from featnames import SLR, BYR


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_paths(**kwargs):
    # log directory
    role = BYR if kwargs[BYR] else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # run id
    run_id = 'entropy_{}'.format(kwargs['entropy_coeff'])
    if 'suffix' in kwargs:
        run_id += '_{}'.format(kwargs['suffix'])

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir
