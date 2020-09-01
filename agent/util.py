import os
import numpy as np
from agent.const import FULL, SPARSE, NOCON
from constants import AGENT_DIR, POLICY_SLR, POLICY_BYR, DROPOUT_GRID
from featnames import SLR, BYR, DROPOUT


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_paths(**kwargs):
    # log directory
    log_dir = get_log_dir(kwargs[BYR])

    # run id
    dropout = DROPOUT_GRID[kwargs[DROPOUT]]
    dropout = [int(10 * dropout[i]) for i in range(len(dropout))]
    run_id = '{}_{}_{}'.format(kwargs['con_set'], dropout[0], dropout[1])

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir


def define_con_set(con_set=None, byr=False):
    if con_set == FULL:
        num_con = 101
    elif con_set == SPARSE:
        num_con = 11
    elif con_set == NOCON:
        num_con = 2
    else:
        raise ValueError('Invalid concession set: {}'.format(con_set))

    cons = np.arange(num_con) / (num_con - 1)
    if not byr:
        cons = np.concatenate([cons, [1.1]])  # expiration rejection
    return cons
