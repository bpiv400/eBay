"""
Utility functions for use in objects related to the RL environment
"""
import torch
import os
import numpy as np
import pandas as pd
from compress_pickle import load, dump
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from constants import (INPUT_DIR, ENV_SIM_DIR, DAY, ARRIVAL_MODELS)
from rlenv.env_consts import (SIM_CHUNKS_DIR, SIM_VALS_DIR, OFFER_MAPS,
                              SIM_DISCRIM_DIR, DATE_FEATS, NORM_IND,
                              LOOKUP, X_LSTG)
from utils import extract_clock_feats, is_split, slr_norm, byr_norm


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
    return proper_squeeze(cat.sample(sample_shape=(1,)).float()).numpy()


def sample_bernoulli(params):
    dist = Bernoulli(logits=params)
    return proper_squeeze(dist.sample((1,))).numpy()


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


def dump_chunk(x_lstg=None, lookup=None, path=None):
    d = {
        X_LSTG: x_lstg,
        LOOKUP: lookup
    }
    dump(d, path)


def align_x_lstg_lookup(x_lstg, lookup):
    x_lstg = pd.concat([df.reindex(index=lookup.index) for df in x_lstg.values()],
                       axis=1)
    return x_lstg


def load_chunk(base_dir=None, num=None, input_path=None):
    """
    Loads a simulator chunk containing x_lstg and lookup
    :param base_dir: base directory of partition
    :param num: number of chunk
    :param input_path: optional path to the chunk
    :return: (pd.Dataframe giving x_lstg, pd.DataFrame giving lookup)
    """
    if input_path is None:
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


def load_featnames(name):
    """
    Loads featnames dictionary for a model
    :param name: str giving name (e.g. hist, con_byr),
     see env_consts.py for model names
    :return: dict
    """
    return load(INPUT_DIR + 'featnames/{}.pkl'.format(name))


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


def compare_input_dicts(model=None, stored_inputs=None, env_inputs=None):
    assert len(stored_inputs) == len(env_inputs)
    for feat_set_name, stored_feats in stored_inputs.items():
        env_feats = env_inputs[feat_set_name]
        feat_eq = torch.lt(torch.abs(torch.add(-stored_feats, env_feats)), 1e-4)
        if not torch.all(feat_eq):
            print('Model input inequality found for {} in {}'.format(model, feat_set_name))
            feat_eq = (~feat_eq.numpy())[0, :]
            feat_eq = np.nonzero(feat_eq)[0]
            featnames = load_featnames(model)
            if 'offer' in feat_set_name:
                featnames = featnames['offer']
            else:
                featnames = featnames[feat_set_name]
            for feat_index in feat_eq:
                print('-- INCONSISTENCY IN {} --'.format(featnames[feat_index]))
                print('stored value = {} | env value = {}'.format(stored_feats[0, feat_index],
                                                                  env_feats[0, feat_index]))
            input("Press Enter to continue...")
            # raise RuntimeError("Environment inputs diverged from true inputs")


def need_msg(con, slr=None):
    if not slr:
        return 0 < con < 1
    else:
        return con < 1


def populate_test_model_inputs(full_inputs=None, value=None):
    inputs = dict()
    for feat_set_name, feat_df in full_inputs.items():
        if value is not None:
            curr_set = full_inputs[feat_set_name].loc[value, :]
        else:
            curr_set = full_inputs[feat_set_name]
        curr_set = curr_set.values
        curr_set = torch.from_numpy(curr_set).float()
        if len(curr_set.shape) == 1:
            curr_set = curr_set.unsqueeze(0)
        inputs[feat_set_name] = curr_set
    return inputs
