"""
Utility functions for use in objects related to the RL environment
"""
import utils
from constants import SLR_PREFIX, BYR_PREFIX, ARRIVAL_PREFIX, INPUT_DIR
from rlenv.interface.model_names import FEED_FORWARD, LSTM_MODELS, ARRIVAL
from simulator.nets import FeedForward, LSTM, RNN


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


def get_model_dir(model_name):
    """
    Helper method that returns the path to a model's directory, given
    the name of that model
    TODO: Update to accommodate new days and secs interface

    :param model_name: str naming model (following naming conventions in rlenv/model_names.py)
    :return: str
    """
    # get pathing names
    if SLR_PREFIX in model_name:
        model_type = SLR_PREFIX
        model_name = model_name.replace('{}_'.format(SLR_PREFIX), '')
    elif BYR_PREFIX in model_name:
        model_type = BYR_PREFIX
        model_name = model_name.replace('{}_'.format(BYR_PREFIX), '')
    else:
        model_type = ARRIVAL_PREFIX

    model_dir = '{}/{}/{}/'.format(MODEL_DIR, model_type, model_name)
    return model_dir


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