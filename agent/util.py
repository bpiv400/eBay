import os
import torch
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


def discount_return_slr(reward=None, done=None, info=None):
    """
    Computes time-discounted sum of future rewards from each
    time-step to the end of the batch. Sum resets where `done`
    is 1. Operations vectorized across all trailing dimensions
    after the first [T,].
    :param tensor reward: slr's normalized gross return.
    :param tensor done: indicator for end of trajectory.
    :param namedtuple info: contains other required feats
    :return tensor return_: time-discounted return.
    """
    dtype = reward.dtype  # cast new tensors to this data type
    T, N = reward.shape  # time steps, number of environments

    # recast
    done = done.type(torch.int)
    max_return = info.max_return.type(dtype)

    # initialize matrix of returns
    return_ = torch.zeros(reward.shape, dtype=dtype)

    # initialize variables that track sale outcomes
    t_done = torch.tensor(1e8, dtype=dtype).expand(N)
    net_value = torch.zeros(N, dtype=dtype)

    # to be later marked valid=False
    censored = torch.zeros(reward.shape, dtype=torch.bool)

    for t in reversed(range(T)):
        # time at end of trajectory
        t_done = t_done * (1 - done[t]) + info.months[t] * done[t]

        # months from focal action until end of trajectory
        months_to_done = (t_done - info.months_last[t]).type(dtype)

        # price paid at sale, less fees
        net_value = net_value * (1 - done[t]) + reward[t] * done[t]
        net_value = torch.min(net_value, max_return[t])  # precision

        # discounted sale proceeds
        return_[t] += net_value

        # offers made after listing sells
        censored[t] = months_to_done < 0

    # normalize
    return_ /= max_return

    return return_, censored


def discount_return_byr(reward=None, done=None, info=None):
    """
    Computes time-discounted sum of future rewards from each
    time-step to the end of the batch. Sum resets where `done`
    is 1. Operations vectorized across all trailing dimensions
    after the first [T,].
    :param tensor reward: slr's normalized gross return.
    :param tensor done: indicator for end of trajectory.
    :param namedtuple info: contains other required feats
    :return tensor return_: time-discounted return.
    """
    dtype = reward.dtype  # cast new tensors to this data type
    T, N = reward.shape  # time steps, number of environments

    # recast
    done = done.type(torch.int)
    max_return = info.max_return.type(dtype)

    # initialize matrix of returns
    return_ = torch.zeros(reward.shape, dtype=dtype)

    # initialize variables that track sale outcomes
    net_value = torch.zeros(N, dtype=dtype)

    for t in reversed(range(T)):
        # value less price paid
        net_value = net_value * (1 - done[t]) + reward[t] * done[t]
        net_value = torch.min(net_value, max_return[t])  # precision
        return_[t] += net_value

    # normalize
    return_ /= max_return

    return return_
