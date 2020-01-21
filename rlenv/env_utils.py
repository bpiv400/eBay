"""
Utility functions for use in objects related to the RL environment
"""
import torch
import os
import numpy as np
import pandas as pd
from compress_pickle import load
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson
from constants import (INPUT_DIR, TOL_HALF,
                       MODEL_DIR, ENV_SIM_DIR, DAY, BYR_PREFIX, SLR_PREFIX)
from model.nets import FeedForward
from rlenv.env_consts import (META_6, META_7, SIM_CHUNKS_DIR, SIM_VALS_DIR,
                              SIM_DISCRIM_DIR, THREAD_MAP, DATE_FEATS, 
                              ARRIVAL_MODELS)
from featnames import *


def load_featnames(name):
    """
    Loads featnames dictionary for a model
    #TODO: extend to include agents
    :param name: str giving name (e.g. hist, con_byr),
     see env_consts.py for model names
    :return: dict
    """
    return load(INPUT_DIR + 'featnames/{}.pkl'.format(name))


def load_sizes(name):
    """
        Loads featnames dictionary for a model
        #TODO: extend to include agents
        :param name: str giving name (e.g. hist, con_byr),
         see env_consts.py for model names
        :return: dict
        """
    return load(INPUT_DIR + 'sizes/{}.pkl'.format(name))


def load_params():
    """
        Loads featnames dictionary for a model
        #TODO: extend to include agents
        :return: dict
        """
    return load(INPUT_DIR + 'params.pkl')


def featname(feat, turn):
    """
    Returns the name of a particular feature for a particular turn
    :param feat: str giving featname
    :param turn: int giving turn num
    :return: str
    """
    return '{}_{}'.format(feat, turn)


def model_str(model_name, byr=False):
    """
    returns the string giving the name of an offer model
    model (used to refer to the model in SimulatorInterface
     and Composer

    :param model_name: str giving base name
    :param byr: boolean indicating whether this is a buyer model
    :return:
    """
    if model_name in ARRIVAL_MODELS:
        return model_name
    if not byr:
        name = '{}_{}'.format(model_name, SLR_PREFIX)
    else:
        name = '{}_{}'.format(model_name, BYR_PREFIX)
    return name


def get_clock_feats(time):
    """
    Gets clock features as np.array given the time relative to START (in seconds since)
    Order of features in output should reflect env_consts.CLOCK_FEATS
    :param time: int giving time in seconds since START
    :return: numpy array containing 8 elements, where index 0 is an indicator
    for holiday, indices 2-6 give indicators for the day of week, index 7
    gives the sine transformation of the time of day as a fraction of the day since midnight,
    and index 8 is an indicator for whether the time is in the afternoon.
    """
    # holiday and day of week indicators
    date_feats = DATE_FEATS[time // DAY]

    # second of day, as fraction
    sec_norm = (time % DAY) / DAY

    # sine transformation
    time_of_day = np.sin(sec_norm * np.pi)

    # afternoon indicator
    afternoon = sec_norm >= 0.5

    # concatenate
    return np.append(date_feats, [time_of_day, afternoon])


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


def sample_categorical(params):
    """
    Samples from a categorical distribution
    :param torch.FloatTensor params: logits of categorical distribution
    :return: 1 dimensional np.array
    """
    cat = Categorical(logits=params)
    return proper_squeeze(cat.sample(sample_shape=(1, )).float()).numpy()


def sample_bernoulli(params):
    dist = Bernoulli(logits=params)
    return proper_squeeze(dist.sample((1, ))).numpy()


def sample_poisson(lnmean):
    return int(Poisson(torch.exp(lnmean)).sample((1,)).squeeze())


def get_split(con):
    """
    Determines whether concession is close enough to 50 to trigger split feature
    :param np.float32 con: concession
    :return: bool
    """
    con = con * 100
    output = 1 if abs(50 - con) < (TOL_HALF * 100) else 0
    return output


def last_norm(sources, turn):
    """
    Grabs the value of norm from 2 turns ago
    :param dict sources: environment sources dictionary
    :param int turn: current turn
    :return: np.float32
    """
    if turn <= 2:
        out = 0.0
    else:
        out = sources[THREAD_MAP][featname(NORM, turn - 2)]
    return out


def prev_norm(sources, turn):
    """
    Grabs the value of norm from last turn
    :param dict sources: environment sources dictionary
    :param int turn: current turn
    :return: np.float32
    """
    if turn == 1:
        out = 0.0
    else:
        out = sources[THREAD_MAP][featname(NORM, turn-1)]
    return out


def load_model(full_name):
    """
    Initialize PyTorch network for some model
    :param str full_name: full name of the model
    :return: torch.nn.Module
    """
    print('loading {}'.format(full_name))
    sizes, params = load_sizes(full_name), load_params()
    model_path = '{}{}.net'.format(MODEL_DIR, full_name)
    net = FeedForward(sizes, params)  # type: torch.nn.Module
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)
    for param in net.parameters(recurse=True):
        param.requires_grad = False
    net.eval()
    return net


def get_cut(meta):
    """
    Computes the value fee
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
    """
    Gets time difference between end and start normalized by unit
    :param int start: start time of an interval
    :param int end: end time of an interval
    :param int unit: normalization factor
    :return: np.FloatArray
    """
    diff = (end - start) / unit
    diff = np.array([diff], dtype=np.float32)
    return diff


def slr_rej(sources, turn, expire=False):
    """
    Returns outcome series associated with a slr expiration or slr
    automatic rejection
    :param dict sources: environment sources dictionary
    :param int turn: turn of rejection
    :param bool expire: whether this is an expiration rej (automatic if not)
    :return: pd.Series
    """
    outcomes = pd.Series(0.0, index=ALL_OUTCOMES[turn])
    outcomes[featname(REJECT, turn)] = 1
    outcomes[featname(NORM, turn)] = last_norm(sources, turn)
    if not expire:
        outcomes[featname(AUTO, turn)] = 1
    else:
        outcomes[featname(EXP, turn)] = 1
    return outcomes


def slr_auto_acc(sources, turn):
    outcomes = pd.Series(0.0, index=ALL_OUTCOMES[turn])
    outcomes[featname(CON, turn)] = 1
    prev_byr_norm = prev_norm(sources, turn)
    outcomes[featname(NORM, turn)] = 1 - prev_byr_norm
    return outcomes


def get_checkpoint_path(part_dir, chunk_num, discrim=False):
    """
    Returns path to checkpoint file for the given chunk
    and environment simulation (discrim or values)
    :param str part_dir: path to root directory of partition
    :param int chunk_num: chunk id
    :param bool discrim: whether this is a discrim input sim
    :return: str
    """
    if discrim:
        insert = 'discrim'
    else:
        insert = 'val'
    return '{}chunks/{}_check_{}.gz'.format(part_dir, chunk_num, insert)


def get_env_sim_dir(part):
    """
    Gets path to root directory of partition
    :param str part: partition in PARTITIONS
    :return: str
    """
    return '{}{}/'.format(ENV_SIM_DIR, part)


def get_env_sim_subdir(part=None, base_dir=None, chunks=False,
                       values=False, discrim=False):
    """
    Returns the path to a chosen data subdirectory of the current
    environment simulation (discrim dir, vals dir, or chunks dir)

    Either give base_dir or part
    :param str part: partition in PARTITIONS
    :param str base_dir: path to base dir of partition
    :param bool chunks: whether to make path to chunk directory
    :param bool values: whether to make path to value estimates directory
    :param bool discrim: whether to make path discrim inputs directory
    :return: str
    :raises RuntimeError: when subdir type is not set explicitly
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
    :param str record_dir: path to directory containing outputs for simulation
    :param int num: chunk number
    """
    return '{}done_{}.txt'.format(record_dir, num)


def load_output_chunk(directory, c):
    """
    Loads all the simulator output pieces for a chunk
    :param str directory: path to records directory
    :param int c: chunk num
    :return: list containing all output pieces (dicts or dataframes)
    """
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
    """
    Loads all simulator outputs for a value estimation or discrim input sim
    and concatenates each set of output dataframes into 1 dataframe
    :param str part: partition in PARTITIONS
    :param bool values: whether this is a value simulation
    :return: {str: pd.Dataframe}
    """
    directory = get_env_sim_subdir(part=part, values=values, discrim=not values)
    chunks = [path for path in os.listdir(directory) if path.isnumeric()]
    output_list = list()
    for chunk in chunks:
        output_list = output_list + load_output_chunk(directory, int(chunk))
    if values:
        output = {'values': pd.concat(output_list, axis=1)}
    else:
        output = dict()
        for dataset in ['offers', 'threads', 'sales']:
            output[dataset] = [piece[dataset] for piece in output_list]
            output[dataset] = pd.concat(output[dataset], axis=1)
    return output


def load_chunk(base_dir, num):
    """
    Loads a simulator chunk containing x_lstg and lookup
    :param base_dir: base directory of partition
    :param num: number of chunk
    :return: (pd.Dataframe giving x_lstg, pd.DataFrame giving lookup)
    """
    input_path = '{}chunks/{}.gz'.format(base_dir, num)
    input_dict = load(input_path)
    x_lstg = input_dict['x_lstg']
    lookup = input_dict['lookup']
    return x_lstg, lookup


def update_byr_outcomes(con=None, delay=None, sources=None, turn=0):
    """
    Creates a new seller outcome vector given the concession and optionally
    given the delay (in cases where Agent predicts con and delay)
    :param np.float con: concession
    :param np.int delay: selected delay
    :param Sources sources: sources object
    :param int turn: turn number
    :return: (outcome series, sample message boolean)
    """
    outcomes = pd.Series(0.0, index=ALL_OUTCOMES[turn])
    # don't sample a msg if the delay is given since this means an agent
    # selects delay
    if delay is not None:
        sample_msg = False
    else:
        sample_msg = (con != 0 and con != 1)
    if sample_msg:
        outcomes[featname(SPLIT, turn)] = get_split(con)
    outcomes[featname(CON, turn)] = con
    prev_slr_norm = prev_norm(sources, turn)
    prev_byr_norm = last_norm(sources, turn)
    norm = (1 - prev_slr_norm) * con + prev_byr_norm * (1 - con)
    outcomes[featname(NORM, turn)] = norm
    return outcomes, sample_msg
    # TODO FIGURE OUT DELAY SHIT LATER IN A SEPARATE FUNCTION


def update_slr_outcomes(con=None, delay=None, sources=None, turn=0):
    """
    Creates a new seller outcome vector given the concession and optionally
    given the delay (in cases where Agent predicts con and delay)
    :param np.float con: concession
    :param np.int delay: selected delay
    :param Sources sources: sources object
    :param int turn: turn number
    :return: (outcome series, sample message boolean)
    """
    outcomes = pd.Series(0.0, index=ALL_OUTCOMES[turn])
    # don't sample a msg if the delay is given since this means an agent
    # selects delay
    if delay is not None:
        sample_msg = False
    else:
        sample_msg = (con != 0 and con != 1)
    # compute previous seller norm or set to 0 if this is the first turn
    prev_slr_norm = last_norm(sources, turn)
    # handle rejection case
    if con == 0:
        outcomes[featname(REJECT, turn)] = 1
        outcomes[featname(NORM, turn)] = prev_slr_norm
    else:
        outcomes[featname(CON, turn)] = con
        outcomes[featname(SPLIT, turn)] = get_split(con)
        prev_byr_norm = sources[THREAD_MAP][featname(NORM, turn - 1)]
        norm = slr_norm(con=con, prev_byr_norm=prev_byr_norm, prev_slr_norm=prev_slr_norm)
        outcomes[featname(NORM, turn)] = norm

    # TODO FIGURE OUT DELAY SHIT LATER IN A SEPARATE FUNCTION

    return outcomes, sample_msg


def slr_norm(con, prev_byr_norm, prev_slr_norm):
    norm = 1 - con * prev_byr_norm - (1 - prev_slr_norm) * (1 - con)
    return norm
