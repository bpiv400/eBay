import os
from constants import AGENT_DIR, POLICY_SLR, POLICY_BYR
from featnames import SLR, BYR


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_paths(byr=None, name=None):
    # log directory
    log_dir = get_log_dir(byr)

    # run id
    run_id = name

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir
