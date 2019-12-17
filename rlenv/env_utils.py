"""
Utility functions for use in objects related to the RL environment
"""
import torch
import numpy as np
import pandas as pd
import utils
from torch.distributions.categorical import Categorical
from constants import (INPUT_DIR, START, HOLIDAYS, TOL_HALF, MODEL_DIR)
from rlenv.interface.model_names import LSTM_MODELS
from rlenv.composer.maps import THREAD_MAP
from models.nets import FeedForward, LSTM
from rlenv.env_consts import (META_6, META_7, DAY, NORM, ALL_OUTCOMES,
                              AUTO, REJECT, EXP)


def load_featnames(full_name):
    featnames_path = '{}featnames/{}.pkl'.format(INPUT_DIR, full_name)
    featnames_dict = utils.unpickle(featnames_path)
    return featnames_dict


def load_sizes(full_name):
    featnames_path = '{}sizes/{}.pkl'.format(INPUT_DIR, full_name)
    featnames_dict = utils.unpickle(featnames_path)
    return featnames_dict


def featname(feat, turn):
    return '{}_{}'.format(feat, turn)


def get_model_class(full_name):
    """
    Returns the class of the given model

    :param full_name: str giving the name of the model
    :return: simulator.nets.RNN, simulator.nets.LSTM, or
    simulator.nets.FeedForward
    """
    if full_name in LSTM_MODELS:
        mod_type = LSTM
    else:
        mod_type = FeedForward
    return mod_type


def get_clock_feats(time):
    """
    Gets clock features as np.array given the time relative to START (in seconds since)

    :param time: int giving time in seconds since START
    :return: torch.FloatTensor containing 8 elements, where the first element is an indicator
    for holiday, the second through seventh give indicators for the day of week, and the 8th
    gives the time of day as a fraction of seconds since minute
    """
    out = np.zeros(8)
    clock = pd.to_datetime(time, unit='s', origin=START)
    # holidays
    out[0] = int(str(clock.date()) in HOLIDAYS)
    # dow number
    if clock.dayofweek < 6:
        out[clock.dayofweek + 1] = 1
    # minutes of day
    out[7] = (clock.hour * 60 * 60 + clock.minute * 60 + clock.second) / DAY
    return out


def proper_squeeze(tensor):
    """
    Squeezes a tensor to 1 rather than 0 dimensions

    :param tensor: torch.tensor with only 1 non-singleton dimension
    :return: 1 dimensional tensor
    """
    tensor = tensor.squeeze()
    if len(tensor.shape) == 0:
        tensor = tensor.unsqueeze(0)
    return tensor


def categorical_sample(params, n):
    cat = Categorical(logits=params)
    return proper_squeeze(cat.sample(sample_shape=(n, )).float()).numpy()


def get_split(con):
    con = con * 100
    output = 1 if abs(50 - con) < (TOL_HALF * 100) else 0
    return output


def last_norm(sources, turn):
    if turn <= 2:
        out = 0.0
    else:
        out = sources[THREAD_MAP][featname(NORM, turn - 2)]
    return out


def prev_norm(sources, turn):
    if turn == 1:
        out = 0.0
    else:
        out = sources[THREAD_MAP][featname(NORM, turn-1)]
    return out


def chunk_dir(part_dir, chunk_num, records=False, rewards=False, discrim=False):
    if records:
        if discrim:
            insert = 'discrim'
        else:
            insert = 'records'
    elif rewards:
        insert = 'rewards'
    else:
        raise RuntimeError('invalid chunk dir params')
    return '{}{}/{}/'.format(part_dir, insert, chunk_num)


def load_model(full_name):
    """
    Initialize pytorch network for some model

    :param full_name: full name of the model
    :return: PyTorch Module
    """
    sizes = load_sizes(full_name)
    model_path = '{}{}.net'.format(MODEL_DIR, full_name)
    model_class = get_model_class(full_name)
    net = model_class(sizes, dropout=False)  # type: torch.nn.Module
    state_dict = torch.load(model_path, map_location='cpu')
    net.load_state_dict(state_dict)
    for param in net.parameters(recurse=True):
        param.requires_grad = False
    net.eval()
    return net


def get_cut(meta):
    """
    Computes the value fee. For now, just set to 10%
    of sale price, pending refinement decisions

    :param meta: integer giving meta category
    :return: float
    """
    rate = .09
    if meta in META_7:
        rate = .07
    elif meta in META_6:
        rate = .06
    return rate


def time_delta(start, end, unit=DAY):
    diff = (end - start) / unit
    diff = np.array([diff], dtype=np.float32)
    return diff


def slr_rej(sources, turn, expire=False):
    outcomes = pd.Series(0.0, index=ALL_OUTCOMES[turn])
    outcomes[featname(REJECT, turn)] = 1
    outcomes[featname(NORM, turn)] = last_norm(sources, turn)
    if not expire:
        outcomes[featname(AUTO, turn)] = 1
    else:
        outcomes[featname(EXP, turn)] = 1
    return outcomes
