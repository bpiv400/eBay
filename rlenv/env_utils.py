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
from constants import (INPUT_DIR, TOL_HALF,
                       MODEL_DIR, ENV_SIM_DIR, DAY, BYR_PREFIX, SLR_PREFIX)
from nets.FeedForward import FeedForward
from rlenv.env_consts import (META_6, META_7, SIM_CHUNKS_DIR, SIM_VALS_DIR, OFFER_MAPS,
                              SIM_DISCRIM_DIR, DATE_FEATS, ARRIVAL_MODELS, NORM_IND)
from featnames import *
from utils import extract_clock_feats, is_split, slr_norm, byr_norm


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

    # concatenate
    return np.append(date_feats, extract_clock_feats(time)).astype('float32')


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


def last_norm(sources=None, turn=0):
    """
    Grabs the value of norm from 2 turns ago
    :param dict sources: environment sources dictionary
    :param int turn: current turn
    :return: np.float32
    """
    if turn <= 2:
        out = 0.0
    else:
        offer_map = OFFER_MAPS[turn - 2]
        out = sources[offer_map][NORM_IND]
    return out


def prev_norm(sources=None, turn=0):
    """
    Grabs the value of norm from last turn
    :param dict sources: environment sources dictionary
    :param int turn: current turn
    :return: np.float32
    """
    if turn == 1:
        out = 0.0
    else:
        offer_map = OFFER_MAPS[turn - 1]
        out = sources[offer_map][NORM_IND]
    return out


def load_model(full_name):
    """
    Initialize PyTorch network for some model
    :param str full_name: full name of the model
    :return: torch.nn.Module
    """
    print('loading {}'.format(full_name))

    # create neural network
    sizes = load_sizes(full_name)
    net = FeedForward(sizes, dropout=False)  # type: torch.nn.Module

    # read in model parameters
    model_path = '{}{}.net'.format(MODEL_DIR, full_name)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # delete dropout parameters
    drop = [k for k in state_dict.keys() if 'log_alpha' in k]
    for k in drop:
        del state_dict[k]

    # load parameters into model
    net.load_state_dict(state_dict)

    # eval mode
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
    :return: np.array
    """
    diff = (end - start) / unit
    diff = np.array([diff], dtype=np.float32)
    return diff[0]


def slr_rej_outcomes(sources, turn):
    """
    Returns outcome series associated with a slr expiration or slr
    automatic rejection
    :param dict sources: environment sources dictionary
    :param int turn: turn of rejection
    :return: np.array
    """
    norm = last_norm(sources=sources, turn=turn)
    return np.array([0.0, 1.0, norm, 0.0, 0.0], dtype=np.float32)


def slr_auto_acc_outcomes(sources, turn):
    """
    Returns offer outcomes (including msg) associated with a slr
    auto acceptance -- ordered according to OFFER_FEATS
    :param sources: source dict
    :param turn: turn number
    :return: np.array
    """
    con = 1.0
    prev_byr_norm = prev_norm(sources=sources, turn=turn)
    norm = 1.0 - prev_byr_norm
    return np.array([con, 0.0, norm, 0.0, 0.0], dtype=np.float32)


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


def get_delay_outcomes(seconds=0, max_delay=0, turn=0):
    """
    Generates all features
    :param seconds: number of seconds delayed
    :param max_delay: max possible duration of delay
    :param turn: turn number
    :return: np.array
    """
    delay = get_delay(seconds, max_delay)
    days = get_days(seconds)
    exp = get_exp(delay, turn)
    auto = get_auto(delay, turn)
    return np.array([days, delay, auto, exp], dtype=np.float)


def get_delay(seconds, max_delay):
    return seconds / max_delay


def get_days(seconds):
    return seconds / DAY


def get_exp(delay, turn):
    if turn % 2 == 0:
        exp = delay == 1
    else:
        exp = 0
    return exp


def get_auto(delay, turn):
    if turn % 2 == 0:
        auto = delay == 0
    else:
        auto = 0
    return auto


def get_con_outcomes(con=None, sources=None, turn=0):
    """
    Returns vector giving con and features downstream from it in order given by
    OUTCOME_FEATS -- Doesn't include msg
    :param con: con
    :param sources: source dictionary
    :param turn: turn number
    :return:
    """
    reject = get_reject(con)
    if turn % 2 == 0:
        prev_byr_norm = prev_norm(sources=sources, turn=turn)
        prev_slr_norm = last_norm(sources=sources, turn=turn)
        norm = slr_norm(con=con, prev_byr_norm=prev_byr_norm, prev_slr_norm=prev_slr_norm)
    else:
        prev_byr_norm = last_norm(sources=sources, turn=turn)
        prev_slr_norm = prev_norm(sources=sources, turn=turn)
        norm = byr_norm(con=con, prev_byr_norm=prev_byr_norm, prev_slr_norm=prev_slr_norm)
    split = is_split(con)
    return np.array([con, reject, norm, split], dtype=np.float)


def get_reject(con):
    return con == 0
