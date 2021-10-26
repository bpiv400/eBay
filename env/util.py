"""
Utility functions for use in objects related to the RL environment
"""
import numpy as np
import torch
from nets.FeedForward import FeedForward
from paths import MODEL_DIR
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from utils import load_sizes
from agent.const import COMMON_CONS
from env.const import OFFER_MAPS, DATE_FEATS_ARRAY, NORM_IND
from constants import DAY, MAX_DELAY_TURN
from featnames import ARRIVAL_MODELS


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
    date_feats = DATE_FEATS_ARRAY[time // DAY]

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


def get_delay_outcomes(seconds=0, turn=0):
    """
    Generates all features
    :param seconds: number of seconds delayed
    :param turn: turn number
    :return: np.array
    """
    delay = seconds / MAX_DELAY_TURN
    days = seconds / DAY
    exp = 0 if turn % 2 == 1 else (delay == 1)
    auto = 0 if turn % 2 == 1 else (delay == 0)
    return np.array([days, delay, auto, exp], dtype=np.float)


def get_con_outcomes(con=None, sources=None, turn=0):
    """
    Returns vector giving con and features downstream from it in order given by
    OUTCOME_FEATS -- Doesn't include msg
    :param con: con
    :param sources: source dictionary
    :param turn: turn number
    :return:
    """
    reject = con == 0
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


def need_msg(con, slr=None):
    if not slr:
        return 0 < con < 1
    else:
        return con < 1


def load_model(name, verbose=False):
    """
    Initialize PyTorch network for some model
    :param str name: full name of the model
    :param bool verbose: print statements if True
    :return: torch.nn.Module
    """
    if verbose:
        print('Loading {} model'.format(name))

    # create neural network
    sizes = load_sizes(name)
    net = FeedForward(sizes)  # type: torch.nn.Module

    # read in model parameters
    path = '{}{}.net'.format(MODEL_DIR, name)
    state_dict = torch.load(path, map_location=torch.device('cpu'))

    # load parameters into model
    net.load_state_dict(state_dict, strict=True)

    # eval mode
    for param in net.parameters(recurse=True):
        param.requires_grad = False
    net.eval()

    # use shared memory
    try:
        net.share_memory()
    except RuntimeError:
        pass

    return net


def slr_norm(con=None, prev_byr_norm=None, prev_slr_norm=None):
    """
    Normalized offer for seller turn.
    :param con: current concession, between 0 and 1.
    :param prev_byr_norm: normalized concession from one turn ago.
    :param prev_slr_norm: normalized concession from two turns ago.
    :return: normalized distance of current offer from start_price to 0.
    """
    return 1 - con * prev_byr_norm - (1 - prev_slr_norm) * (1 - con)


def byr_norm(con=None, prev_byr_norm=None, prev_slr_norm=None):
    """
    Normalized offer for buyer turn.
    :param con: current concession, between 0 and 1.
    :param prev_byr_norm: normalized concession from two turns ago.
    :param prev_slr_norm: normalized concession from one turn ago.
    :return: normalized distance of current offer from 0 to start_price.
    """
    return (1 - prev_slr_norm) * con + prev_byr_norm * (1 - con)


def extract_clock_feats(seconds):
    """
    Creates clock features from timestamps.
    :param seconds: seconds since START.
    :return: tuple of time_of_day sine transform and afternoon indicator.
    """
    sec_norm = (seconds % DAY) / DAY
    time_of_day = np.sin(sec_norm * np.pi)
    afternoon = sec_norm >= 0.5
    return time_of_day, afternoon