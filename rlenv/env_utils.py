"""
Utility functions for use in objects related to the RL environment
"""
import torch
import pandas as pd
import utils
from torch.distributions.categorical import Categorical
from constants import (INPUT_DIR, START,
                       HOLIDAYS, TOL_HALF, MODEL_DIR)
from rlenv.interface.model_names import LSTM_MODELS
from simulator.nets import FeedForward, LSTM
from rlenv.env_consts import PARAMS_PATH, META_6, META_7, DAY


def load_featnames(full_name):
    featnames_path = '{}featnames/{}.pkl'.format(INPUT_DIR, full_name)
    featnames_dict = utils.unpickle(featnames_path)
    return featnames_dict


def load_sizes(full_name):
    featnames_path = '{}sizes/{}.pkl'.format(INPUT_DIR, full_name)
    featnames_dict = utils.unpickle(featnames_path)
    return featnames_dict


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
    out = torch.zeros(8).float()
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
    return proper_squeeze(cat.sample(sample_shape=(n, )).float())


def get_split(con):
    con = con * 100
    output = 1 if abs(50 - con) < (TOL_HALF * 100) else 0
    return output


def load_params(model_exp):
    df = pd.read_csv(PARAMS_PATH, index_col='id')
    params = df.loc[model_exp].to_dict()
    return params


def load_model(full_name, model_exp):
    """
    Initialize pytorch network for some model

    :param full_name: full name of the model
    :param model_exp: experiment number for the model
    :return: PyTorch Module
    """
    sizes = load_sizes(full_name)
    params = load_params(model_exp)
    model_path = '{}{}_{}.net'.format(MODEL_DIR, full_name, model_exp)
    # loading model
    model_class = get_model_class(full_name)
    net = model_class(params, sizes)  # type: torch.nn.Module
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    return net
    # except (RuntimeError, FileNotFoundError) as e:
    # print(e)
    # print('failed for {}'.format(err_name))
    # return None


def get_value_fee(price, meta):
    """
    Computes the value fee. For now, just set to 10%
    of sale price, pending refinement decisions

    # TODO: What did we decide about shipping?

    :param price: price of sale
    :param meta: integer giving meta category
    :return: float
    """
    rate = .09
    if meta in META_7:
        rate = .07
    elif meta in META_6:
        rate = .06
    return rate * price


def time_delta(start, end, unit=DAY):
    diff = (end - start) / unit
    diff = torch.tensor([diff]).float()
    return diff