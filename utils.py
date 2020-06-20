import argparse
import pickle
import torch
from torch.nn.functional import log_softmax
import numpy as np
from compress_pickle import load
from nets.FeedForward import FeedForward
from constants import DAY, MONTH, SPLIT_PCTS, INPUT_DIR, \
    MODEL_DIR, META_6, META_7, LISTING_FEE, PARTITIONS, PARTS_DIR, \
    MAX_DELAY_TURN, MAX_DELAY_ARRIVAL, EMPTY


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    return pickle.load(open(file, "rb"))


def get_remaining(lstg_start, delay_start):
    """
    Calculates number of delay intervals remaining in lstg.
    :param lstg_start: seconds from START to start of lstg.
    :param delay_start: seconds from START to beginning of delay window.
    """
    remaining = lstg_start - delay_start
    remaining += MAX_DELAY_ARRIVAL
    remaining /= MAX_DELAY_TURN
    remaining = np.minimum(1.0, remaining)
    return remaining


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


def is_split(con):
    """
    Boolean for whether concession is (close to) an even split.
    :param con: scalar or Series of concessions.
    :return: boolean or Series of booleans.
    """
    return con in SPLIT_PCTS


def get_months_since_lstg(lstg_start=None, time=None):
    """
    Float number of months between inputs.
    :param lstg_start: seconds from START to lstg start.
    :param time: seconds from START to focal event.
    :return: number of months between lstg_start and start.
    """
    return (time - lstg_start) / MONTH


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


def load_sizes(name):
    """
    Loads featnames dictionary for a model
    :param name: str giving name (e.g. hist, con_byr),
     see const.py for model names
    :return: dict
    """
    return load(INPUT_DIR + 'sizes/{}.pkl'.format(name))


def load_featnames(name):
    """
    Loads featnames dictionary for a model
    #TODO: extend to include agents
    :param name: str giving name (e.g. hist, con_byr),
     see const.py for model names
    :return: dict
    """
    return load(INPUT_DIR + 'featnames/{}.pkl'.format(name))


def load_state_dict(name=None):
    """
    Loads state dict of a model
    :param name: string giving name of model (see consts)
    :return: dict
    """
    model_path = '{}{}.net'.format(MODEL_DIR, name)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    return state_dict


def load_model(name, verbose=False):
    """
    Initialize PyTorch network for some model
    :param str name: full name of the model
    :param verbose: boolean for printing statements
    :return: torch.nn.Module
    """
    if verbose:
        print('Loading {} model'.format(name))

    # create neural network
    sizes = load_sizes(name)
    net = FeedForward(sizes)  # type: nn.Module

    if not EMPTY:
        # read in model parameters
        state_dict = load_state_dict(name=name)

        # load parameters into model
        net.load_state_dict(state_dict, strict=True)

    # eval mode
    for param in net.parameters(recurse=True):
        param.requires_grad = False
    net.eval()

    return net


def get_cut(meta):
    if meta in META_6:
        return .06
    if meta in META_7:
        return .07
    return .09


def slr_reward(months_to_sale=None, months_since_start=None,
               sale_proceeds=None, monthly_discount=None,
               action_diff=None, action_discount=None, action_cost=None):
    """
    Discounts proceeds from sale and listing fees paid.
    :param months_to_sale: months from listing start to sale
    :param months_since_start: months since start of listing
    :param sale_proceeds: sale price net of eBay cut
    :param monthly_discount: multiplicative factor on proceeds, by month
    :param action_diff: number of actions from current state until sale
    :param action_discount: multiplicative factor of proceeds, by action
    :param action_cost: cost per action
    :return: discounted net proceeds
    """
    # discounted listing fees
    M = np.ceil(months_to_sale) - np.ceil(months_since_start) + 1
    if monthly_discount is not None:
        k = months_since_start % 1
        factor = (1 - monthly_discount ** M) / (1 - monthly_discount)
        delta = (monthly_discount ** (1-k)) * factor
        costs = LISTING_FEE * delta
    else:
        costs = LISTING_FEE * M
    # add in action costs
    if action_diff is not None and action_cost is not None:
        costs += action_cost * action_diff
    # discounted proceeds
    if monthly_discount is not None:
        months_diff = months_to_sale - months_since_start
        assert (months_diff >= 0).all()
        sale_proceeds *= monthly_discount ** months_diff
    if action_diff is not None and action_discount is not None:
        sale_proceeds *= action_discount ** action_diff
    return sale_proceeds - costs


def max_slr_reward(months_since_start=None, bin_proceeds=None,
                   monthly_discount=None):
    """
    Discounts proceeds from sale and listing fees paid.
    :param months_since_start: months since start of listing
    :param bin_proceeds: start price net of eBay cut
    :param monthly_discount: multiplicative factor on proceeds, by month
    :return: discounted maximum proceeds
    """
    # discounted listing fees
    if monthly_discount is not None:
        k = months_since_start % 1
        costs = LISTING_FEE * (monthly_discount ** (1-k))
    else:
        costs = LISTING_FEE
    return bin_proceeds - costs


def byr_reward(net_value=None, months_diff=None,
               monthly_discount=None, action_diff=None,
               action_discount=None, action_cost=None):
    """
    Discounts proceeds from sale and listing fees paid.
    :param net_value: value less price paid; 0 if no sale
    :param months_diff: months until purchase; np.inf if no sale
    :param monthly_discount: multiplicative factor on proceeds, by month
    :param action_diff: number of actions from current state until sale
    :param action_discount: multiplicative factor of proceeds, by action
    :param action_cost: cost per action
    :return: discounted net proceeds
    """
    if monthly_discount is not None:
        net_value *= monthly_discount ** months_diff
    if action_discount is not None:
        net_value *= action_discount ** action_diff
    if action_cost is not None:
        net_value -= action_cost * action_diff
    return net_value


def get_model_predictions(m, x):
    """
    Returns predicted categorical distribution.
    :param str m: name of model
    :param dict x: dictionary of input tensors
    :return: torch tensor
    """
    # initialize neural net
    net = load_model(m, verbose=False)
    if torch.cuda.is_available():
        net = net.to('cuda')

    # split into batches
    v = np.array(range(len(x['lstg'])))
    batches = np.array_split(v, 1 + len(v) // 2048)

    # model predictions
    p0 = []
    for b in batches:
        x_b = {k: torch.from_numpy(v[b, :]) for k, v in x.items()}
        if torch.cuda.is_available():
            x_b = {k: v.to('cuda') for k, v in x_b.items()}
        theta_b = net(x_b).cpu().double()
        p0.append(np.exp(log_softmax(theta_b, dim=-1)))

    # concatenate and return
    return torch.cat(p0, dim=0).numpy()


def input_partition():
    """
    Parses command line input for partition name.
    :return part: string partition name.
    """
    parser = argparse.ArgumentParser()

    # partition
    parser.add_argument('--part', required=True, type=str,
                        choices=PARTITIONS, help='partition name')
    return parser.parse_args().part


def init_optional_arg(kwargs=None, name=None, default=None):
    if name not in kwargs:
        kwargs[name] = default


def load_file(part, x):
    """
    Loads file from partitions directory.
    :param str part: name of partition
    :param x: name of file
    :return: dataframe
    """
    return load(PARTS_DIR + '{}/{}.gz'.format(part, x))


def init_x(part, idx=None):
    """
    Initialized dictionary of input dataframes.
    :param str part: name of partition
    :param idx: (multi-)index to reindex with
    :return: dictionary of (reindexed) input dataframes
    """
    x = load_file(part, 'x_lstg')
    x = {k: v.astype('float32') for k, v in x.items()}
    if idx is not None:
        if len(idx.names) == 1:
            x = {k: v.reindex(index=idx) for k, v in x.items()}
        else:
            x = {k: v.reindex(index=idx, level='lstg')
                 for k, v in x.items()}
    return x
