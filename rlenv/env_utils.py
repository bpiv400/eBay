"""
Utility functions for use in objects related to the RL environment
"""
import torch
import pandas as pd
import utils
from torch.distributions.categorical import Categorical
from constants import SLR_PREFIX, BYR_PREFIX, ARRIVAL_PREFIX, INPUT_DIR, START, HOLIDAYS, TOL_HALF
from rlenv.interface.model_names import FEED_FORWARD, LSTM_MODELS, ARRIVAL
from simulator.nets import FeedForward, LSTM, RNN
from rlenv.env_consts import PARAMS_PATH


def model_str(model_name, byr=False):
    """
    returns the string giving the name of an offer model
    model (used to refer to the model in SimulatorInterface
     and Composer

    :param model_name: str giving base name
    :param byr: boolean indicating whether this is a buyer model
    :return:
    """
    if not byr:
        name = '{}_{}'.format(SLR_PREFIX, model_name)
    else:
        name = '{}_{}'.format(BYR_PREFIX, model_name)
    return name


def get_model_type(full_name):
    if full_name in ARRIVAL:
        model_type = ARRIVAL_PREFIX
    elif SLR_PREFIX in full_name:
        model_type = SLR_PREFIX
    else:
        model_type = BYR_PREFIX
    return model_type


def get_model_name(full_name):
    if full_name in ARRIVAL:
        model_name = full_name
    elif SLR_PREFIX in full_name:
        model_name = full_name.replace('{}_'.format(SLR_PREFIX), '')
    else:
        model_name = full_name.replace('{}_'.format(BYR_PREFIX), '')
    return model_name


def load_featnames(model_type, model_name):
    if model_type == ARRIVAL_PREFIX:
        path_suffix = model_name
    else:
        path_suffix = '{}_{}'.format(model_name, model_type)
    featnames_path = '{}featnames/{}.pkl'.format(INPUT_DIR, path_suffix)
    featnames_dict = utils.unpickle(featnames_path)
    return featnames_dict


def load_sizes(model_type, model_name):
    if model_type == ARRIVAL_PREFIX:
        path_suffix = model_name
    else:
        path_suffix = '{}_{}'.format(model_name, model_type)
    featnames_path = '{}sizes/{}.pkl'.format(INPUT_DIR, path_suffix)
    featnames_dict = utils.unpickle(featnames_path)
    return featnames_dict


def get_model_class(model_name):
    """
    Returns the class of the given model
    TODO: Update to accommodate new days and secs interface

    :param model_name: str giving the name of the model
    :return: simulator.nets.RNN, simulator.nets.LSTM, or
    simulator.nets.FeedForward
    """
    if model_name in FEED_FORWARD:
        mod_type = FeedForward
    elif model_name in LSTM_MODELS:
        mod_type = LSTM
    else:
        mod_type = RNN
    return mod_type


def get_featnames_sizes(full_name=None, model_type=None, model_name=None):
    if full_name is not None:
        model_type, model_name = get_model_type(full_name), get_model_name(full_name)
    return load_featnames(model_type, model_name), load_sizes(model_type, model_name)


def get_clock_feats(time):
    """
    Gets clock features as np.array given the time

    For arrival interface, gives outputs in order of ARRIVAL_CLOCK_FEATS
    For offer interface, gives outputs in order of

    Will need to add argument to include minutes for other interface

    :param time: int giving time in seconds since START
    :return: NA
    """
    out = torch.zeros(8, dtype=torch.float64)
    clock = pd.to_datetime(time, unit='s', origin=START)
    # holidays
    out[0] = int(str(clock.date()) in HOLIDAYS)
    # dow number
    if clock.dayofweek < 6:
        out[clock.dayofweek + 1] = 1
    # minutes of day
    out[7] = (clock.hour * 60 * 60 + clock.minute * 60 + clock.second) / DAY
    return out


def get_model_input_paths(model_dir, exp):
    """
    Helper method that returns the paths to files related to some model, given
    that model's path and experiment number

    :param model_dir: string giving path to model directory
    :param exp: int giving integer number of the experiment
    :return: 3-tuple of params path, sizes path, model path
    """
    params_path = '{}params.csv'.format(model_dir)
    sizes_path = '{}sizes.pkl'.format(model_dir)
    model_path = '{}{}.pt'.format(model_dir, exp)
    return params_path, sizes_path, model_path


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
    return cat.sample(sample_shape=(n, )).float()


def get_split(con):
    output = 1 if abs(.5 - con) < TOL_HALF else 0
    return output


def load_params(model_exp):
    df = pd.read_csv(PARAMS_PATH, index_col='id')
    params = df.loc[model_exp].to_dict()
    return params


def load_model(model_type, model_name, model_exp):
        """
        Initialize pytorch network for some model

        :param model_type: type of model (byr, slr, arrival)
        :param model_name: name of the model
        :param model_exp: experiment number for the model
        :return: PyTorch Module
        """
        sizes = load_sizes(model_type, model_name)
        params = load_params(model_exp)
        model_path = 
        try:
            sizes = utils.unpickle(sizes_path)
            params = pd.read_csv(params_path, index_col='id')
            params = params.loc[model_exp].to_dict()
            model_class = get_model_class(model_name)
            net = model_class(params, sizes)
            net.load_state_dict(torch.load(model_path))
        except (RuntimeError, FileNotFoundError) as e:
            print(e)
            print('failed for {}'.format(err_name))
            return None
        return net
