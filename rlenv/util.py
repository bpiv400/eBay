"""
Utility functions for use in objects related to the RL environment
"""
import numpy as np
import pandas as pd
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from agent.const import COMMON_CONS
from constants import INPUT_DIR
from featnames import LOOKUP, X_LSTG, ARRIVALS, ARRIVAL_MODELS
from constants import PARTS_DIR, DAY, MAX_DELAY_TURN
from rlenv.const import (SIM_CHUNKS_DIR, SIM_VALS_DIR, OFFER_MAPS,
                         SIM_DISCRIM_DIR, DATE_FEATS_DF, NORM_IND)
from utils import unpickle, extract_clock_feats, slr_norm, byr_norm


def model_str(model_name, turn=None):
    """
    returns the string giving the name of an offer model
    model (used to refer to the model in SimulatorInterface
     and Composer

    :param model_name: str giving base name
    :param turn: int giving the turn associated with the model
    :return:
    """
    if model_name in ARRIVAL_MODELS:
        return model_name
    else:
        return '{}{}'.format(model_name, turn)


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
    date_feats = DATE_FEATS_DF[time // DAY]

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


def sample_categorical(logits=None, probs=None):
    """
    Samples from a categorical distribution
    :param logits: logits of categorical distribution
    :param probs: probabilities of categorical distribution
    :return: 1 dimensional np.array
    """
    assert logits is None or probs is None
    if logits is not None:
        cat = Categorical(logits=logits)
    else:
        cat = Categorical(probs=probs)
    sample = proper_squeeze(cat.sample(sample_shape=(1,)).float())
    return sample.numpy()


def sample_bernoulli(params):
    dist = Bernoulli(logits=params)
    sample = proper_squeeze(dist.sample((1,)))
    return sample.numpy()


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
    return np.array([0.0, 1.0, norm, 0.0], dtype=np.float32)


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
    return np.array([con, 0.0, norm, 0.0], dtype=np.float32)


def get_env_sim_dir(part):
    """
    Gets path to root directory of partition
    """
    return PARTS_DIR + '{}/'.format(part)


def get_env_sim_subdir(part=None, base_dir=None, chunks=False,
                       values=False, discrim=False):
    """
    Returns the path to a chosen data subdirectory of the current
    environment simulation (discriminator dir, vals dir, or chunks dir)

    Either give base_dir or part
    :param str part: partition in PARTITIONS
    :param str base_dir: path to base dir of partition
    :param bool chunks: whether to make path to chunk directory
    :param bool values: whether to make path to value estimates directory
    :param bool discrim: whether to make path discriminator inputs directory
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


def align_x_lstg_lookup(x_lstg, lookup):
    x_lstg = pd.concat([df.reindex(index=lookup.index) for df in x_lstg.values()],
                       axis=1)
    return x_lstg


def load_chunk(part=None, base_dir=None, num=None, input_path=None):
    """
    Loads a simulator chunk containing x_lstg and lookup
    :param str part: name of partition
    :param base_dir: base directory of partition
    :param num: number of chunk
    :param input_path: optional path to the chunk
    :return: (pd.Dataframe giving x_lstg, pd.DataFrame giving lookup)
    """
    if input_path is None:
        if base_dir is None:
            base_dir = PARTS_DIR + '{}/'.format(part)
        input_path = base_dir + 'chunks/{}.pkl'.format(num)
    input_dict = unpickle(input_path)
    return [input_dict[k] for k in [X_LSTG, LOOKUP, ARRIVALS]]


def get_delay_outcomes(seconds=0, turn=0):
    """
    Generates all features
    :param seconds: number of seconds delayed
    :param turn: turn number
    :return: np.array
    """
    delay = get_delay(seconds)
    days = get_days(seconds)
    exp = get_exp(delay, turn)
    auto = get_auto(delay, turn)
    return np.array([days, delay, auto, exp], dtype=np.float)


def get_delay(seconds):
    return seconds / MAX_DELAY_TURN


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


def load_featnames(name):
    """
    Loads featnames dictionary for a model
    :param name: str giving name (e.g. hist, con_byr),
     see const.py for model names
    :return: dict
    """
    return unpickle(INPUT_DIR + 'featnames/{}.pkl'.format(name))


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
    if turn == 7:
        common = False
    else:
        common = con in COMMON_CONS[turn]
    return np.array([con, reject, norm, common], dtype=np.float)


def get_reject(con):
    return con == 0


def need_msg(con, slr=None):
    if not slr:
        return 0 < con < 1
    else:
        return con < 1
