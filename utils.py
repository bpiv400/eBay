import argparse
import os
import pickle
import pandas as pd
import torch
import numpy as np
from constants import DAY, MAX_DELAY_TURN, MAX_DELAY_ARRIVAL, OUTCOME_SIMS
from paths import PARTS_DIR, SIM_DIR, PCTILE_DIR, INPUT_DIR
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


def get_days_since_lstg(lstg_start=None, time=None):
    """
    Float number of days between inputs.
    :param lstg_start: seconds from START to lstg start.
    :param time: seconds from START to focal event.
    :return: number of days between lstg_start and start.
    """
    return (time - lstg_start) / DAY


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


def load_chunk(part=None, num=None):
    """
    Loads a simulator chunk containing x_lstg and lookup
    :param str part: name of partition
    :param int num: number of chunk
    :return: (pd.Dataframe giving x_lstg, pd.DataFrame giving lookup)
    """
    path = PARTS_DIR + '{}/chunks/{}.pkl'.format(part, num)
    return unpickle(path)


def verify_path(path=None):
    elem = path.split('/')
    folder = '/'.join(elem[:-1])
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if os.path.isfile(path):
        print(f'{path} already exists.')
        exit(0)
