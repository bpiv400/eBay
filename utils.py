import argparse
import os
import pickle
import pandas as pd
import torch
import numpy as np
from nets.FeedForward import FeedForward
from constants import DAY, MAX_DELAY_TURN, MAX_DELAY_ARRIVAL, OUTCOME_SIMS
from paths import PARTS_DIR, SIM_DIR, PCTILE_DIR, FEATS_DIR, MODEL_DIR, INPUT_DIR
from featnames import LOOKUP, X_THREAD, X_OFFER, CLOCK, BYR, SLR, AGENT_PARTITIONS, \
    PARTITIONS, LSTG, SIM, TEST


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj


def topickle(contents=None, path=None):
    """
    Pickles a .pkl file encoded with Python 3
    :param contents: pickle-able object
    :param str path: path to file
    :return: contents of file
    """
    with open(path, "wb") as f:
        pickle.dump(contents, f, protocol=4)


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


def get_days_since_lstg(lstg_start=None, time=None):
    """
    Float number of days between inputs.
    :param lstg_start: seconds from START to lstg start.
    :param time: seconds from START to focal event.
    :return: number of days between lstg_start and start.
    """
    return (time - lstg_start) / DAY


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
    return unpickle(INPUT_DIR + 'sizes/{}.pkl'.format(name))


def load_featnames(name):
    """
    Loads featnames dictionary for a model
    :param name: str giving name (e.g. hist, con_byr),
     see const.py for model names
    :return: dict
    """
    return unpickle(INPUT_DIR + 'featnames/{}.pkl'.format(name))


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


def input_partition(agent=False, opt_arg=None):
    """
    Parses command line input for partition name (and optional argument).
    :param bool agent: if True, raise error when 'sim' partition is called
    :param str opt_arg: optional boolean argument
    :return part: string partition name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True, type=str)
    if opt_arg is not None:
        parser.add_argument('--{}'.format(opt_arg), action='store_true')
    args = parser.parse_args()
    if agent:
        assert args.part in AGENT_PARTITIONS
    else:
        assert args.part in PARTITIONS
    if opt_arg is None:
        return args.part
    else:
        args = vars(args)
        return args['part'], args[opt_arg]


def load_file(part, x, folder=PARTS_DIR):
    """
    Loads file from partitions directory.
    :param str part: name of partition
    :param str x: name of file
    :param str folder: name of folder
    :return: dataframe
    """
    if part in folder:
        path = folder + '{}.pkl'.format(x)
    else:
        path = folder + '{}/{}.pkl'.format(part, x)
    if not os.path.isfile(path):
        return None
    return unpickle(path)


def load_data(part=TEST, sim=False, sim_dir=None, lstgs=None, clock=False, lookup=True):
    if not sim and sim_dir is None:
        folder = PARTS_DIR
    elif sim:
        assert sim_dir is None
        folder = SIM_DIR
    else:
        folder = sim_dir

    # initialize dictionary with lookup file
    data = {}
    if lookup:
        data[LOOKUP] = load_file(part, LOOKUP)
        if sim or sim_dir is not None:
            idx = pd.MultiIndex.from_product([data[LOOKUP].index, range(OUTCOME_SIMS)],
                                             names=[LSTG, SIM])
            data[LOOKUP] = safe_reindex(data[LOOKUP], idx=idx)

    # load other components
    keys = [X_THREAD, X_OFFER]
    if clock:
        keys += [CLOCK]
    for k in keys:
        df = load_file(part, k, folder=folder)
        if df is not None:
            data[k] = df

    # restrict to lstgs
    if lstgs is not None:
        for k, df in data.items():
            data[k] = safe_reindex(df, idx=lstgs)

    return data


def set_gpu(gpu=None):
    """
    Sets the GPU index and the CPU affinity.
    :param int gpu: index of cuda device.
    """
    torch.cuda.set_device(gpu)
    print('Using cuda:{}'.format(gpu))


def get_role(byr=None):
    return BYR if byr else SLR


def safe_reindex(obj=None, idx=None, fill_value=None, dropna=False):
    if type(obj) is dict:
        obj = obj.copy()
        for k, v in obj.items():
            obj[k] = safe_reindex(v, idx=idx, fill_value=fill_value, dropna=dropna)
        return obj

    dtypes = obj.dtypes.to_dict() if dropna else None
    obj = pd.DataFrame(index=idx).join(obj)
    if fill_value is not None:
        obj.loc[obj.isna().squeeze()] = fill_value
    elif dropna:
        obj = obj.loc[~obj.isna().max(axis=1)].astype(dtypes)
    assert obj.isna().sum().sum() == 0

    if len(obj.columns) == 1:
        obj = obj.squeeze()

    return obj


def load_feats(name, lstgs=None, fill_zero=False):
    """
    Loads dataframe of features (and reindexes).
    :param str name: filename
    :param lstgs: listings to restrict to
    :param bool fill_zero: fill missings with 0's if True
    :return: dataframe of features
    """
    df = unpickle(FEATS_DIR + '{}.pkl'.format(name))
    if lstgs is None:
        return df
    kwargs = {'index': lstgs}
    if len(df.index.names) > 1:
        kwargs['level'] = LSTG
    if fill_zero:
        kwargs['fill_value'] = 0.
    return df.reindex(**kwargs)


def load_inputs(part=None, name=None):
    """
    Loads model inputs file from inputs directory.
    :param str part: name of partition
    :param str name: name of file
    :return: dataframe
    """
    path = INPUT_DIR + '{}/{}.pkl'.format(part, name)
    if not os.path.isfile(path):
        return None
    return unpickle(path)


def load_pctile(name=None):
    """
    Loads the percentile file given by name.
    :param str name: name of feature
    :return: pd.Series with feature values in the index and percentiles in the values
    """
    path = PCTILE_DIR + '{}.pkl'.format(name)
    return unpickle(path)


def feat_to_pctile(s=None, pc=None):
    """
    Converts byr hist counts to percentiles or visa versa.
    :param pandas.Series s: counts
    :param pandas.Series pc: percentiles
    :return: Series
    """
    if pc is None:
        pc = load_pctile(name=str(s.name))
    v = pc.reindex(index=s.values, method='pad').values
    return pd.Series(v, index=s.index, name=s.name)


def load_chunk(part=None, num=None):
    """
    Loads a simulator chunk containing x_lstg and lookup
    :param str part: name of partition
    :param int num: number of chunk
    :return: (pd.Dataframe giving x_lstg, pd.DataFrame giving lookup)
    """
    path = PARTS_DIR + '{}/chunks/{}.pkl'.format(part, num)
    return unpickle(path)
