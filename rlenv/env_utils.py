"""
Utility functions for use in objects related to the RL environment
"""
import torch
import os
import numpy as np
import pandas as pd
from compress_pickle import load
import utils
from torch.distributions.categorical import Categorical
from constants import (INPUT_DIR, START, HOLIDAYS, TOL_HALF,
                       MODEL_DIR, ENV_SIM_DIR)
from rlenv.interface.model_names import LSTM_MODELS
from models.nets import FeedForward, LSTM
from rlenv.env_consts import (META_6, META_7, DAY, NORM, ALL_OUTCOMES,
                              AUTO, REJECT, EXP, SIM_CHUNKS_DIR,
                              SIM_VALS_DIR, SIM_DISCRIM_DIR, THREAD_MAP)


def load_featnames(name):
    path = '{}featnames/{}.pkl'.format(INPUT_DIR, name)
    return utils.unpickle(path)


def load_sizes(name):
    path = '{}sizes/{}.pkl'.format(INPUT_DIR, name)
    sizes = utils.unpickle(path)
    if name == 'arrival':
        sizes['out'] = 1
    return sizes


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


def get_chunk_dir(part_dir, chunk_num, discrim=False):
    """
    Gets the path to the data directory containing output files
    for the given environment simulation chunk
    """
    if discrim:
        subdir = get_env_sim_subdir(base_dir=part_dir, discrim=True)
    else:
        subdir = get_env_sim_subdir(base_dir=part_dir, values=True)
    return '{}{}/'.format(subdir, chunk_num)


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


def get_checkpoint_path(part_dir, chunk_num, discrim=False):
    """
    Returns path to checkpoint file for the given chunk
    and environment simulation (discrim or values)
    """
    if discrim:
        insert = 'discrim'
    else:
        insert = 'val'
    return '{}chunks/{}_check_{}.gz'.format(part_dir, chunk_num, insert)


def get_env_sim_dir(part):
    return '{}{}/'.format(ENV_SIM_DIR, part)


def get_env_sim_subdir(part=None, base_dir=None, chunks=False,
                       values=False, discrim=False):
    """
    Returns the path to a chosen data subdirectory of the current
    environment simulation
    """
    if part is not None:
        base_dir = get_env_sim_dir(part)
    if chunks:
        subdir = SIM_CHUNKS_DIR
    elif values:
        subdir = SIM_VALS_DIR
    elif discrim:
        subdir = SIM_DISCRIM_DIR
    else:
        raise RuntimeError("No subdir specified")
    return '{}{}/'.format(base_dir, subdir)


def get_done_file(record_dir, num):
    """
    Generates path to the done file for the given records path and
    chunk number
    """
    return '{}done_{}.txt'.format(record_dir, num)


def load_output_chunk(directory, c):
    subdir = '{}{}/'.format(directory, c)
    pieces = os.listdir(subdir)
    print(subdir)
    print(pieces)
    pieces = [piece for piece in pieces if '.gz' in piece]
    output_list = list()
    for piece in pieces:
        piece_path = '{}{}'.format(subdir, piece)
        output_list.append(load(piece_path))
    return output_list


def load_sim_outputs(part, values=False):
    directory = get_env_sim_subdir(part=part, values=values, discrim=not values)
    chunks = [path for path in os.listdir(directory) if path.isnumeric()]
    output_list = list()
    for chunk in chunks:
        output_list = output_list + load_output_chunk(directory, chunk)
    if values:
        output = {'values': pd.concat(output_list, axis=1)}
    else:
        output = dict()
        for dataset in ['offers', 'threads', 'sales']:
            output[dataset] = [piece[dataset] for piece in output_list]
            output[dataset] = pd.concat(output[dataset], axis=1)
    return output

