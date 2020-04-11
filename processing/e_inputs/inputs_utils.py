import numpy as np
import pandas as pd
from compress_pickle import dump
from processing.processing_utils import load_file
from processing.processing_consts import *
from constants import *
from featnames import *


def get_arrival_times(lstg_start, thread_start, lstg_end=None):
    # thread 0: start of listing
    s = lstg_start.to_frame().assign(thread=0).set_index(
        'thread', append=True).squeeze()

    # threads 1 to N: real threads
    threads = thread_start.reset_index('thread').drop(
        'clock', axis=1).squeeze().groupby('lstg').max().reindex(
        index=lstg_start.index, fill_value=0)

    # thread N+1: end of lstg
    if lstg_end is not None:
        s1 = lstg_end.to_frame().assign(thread=threads + 1).set_index(
            'thread', append=True).squeeze()
        s = pd.concat([s, s1], axis=0)

    # concatenate and sort into single series
    clock = pd.concat([s, thread_start], axis=0).sort_index()

    # thread to int
    idx = clock.index
    clock.index.set_levels(idx.levels[-1].astype('int16'), level=-1, inplace=True)

    return clock.rename('clock')


def save_featnames(x, name):
    """
    Creates dictionary of input feature names.
    :param x: dictionary of input dataframes.
    :param name: string name of model.
    """
    # initialize featnames dictionary
    featnames = {k: list(v.columns) for k, v in x.items() if 'offer' not in k}

    # for offer models
    if 'offer1' in x:
        # buyer models do not have time feats
        if name == 'init_byr' or name[-1] in [str(i) for i in IDX[BYR_PREFIX]]:
            feats = CLOCK_FEATS + OUTCOME_FEATS
        else:
            feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS

        # add turn indicators for RL initializations
        if name.startswith('init'):
            feats += TURN_FEATS[name]

        # check that all offer groupings have same organization
        for k in x.keys():
            if 'offer' in k:
                assert list(x[k].columns) == feats

        # one vector of featnames for offer groupings
        featnames['offer'] = feats

    dump(featnames, INPUT_DIR + 'featnames/{}.pkl'.format(name))


def save_sizes(x, name):
    """
    Creates dictionary of input sizes.
    :param x: dictionary of input dataframes.
    :param name: string name of model.
    """
    sizes = dict()

    # count components of x
    sizes['x'] = {k: len(v.columns) for k, v in x.items()}

    # save interval and interval counts
    if 'arrival' in name:
        sizes['interval'] = INTERVAL[1]
        sizes['interval_count'] = INTERVAL_COUNTS[1]
    elif name.startswith('delay'):
        turn = int(name[-1])
        sizes['interval'] = INTERVAL[turn]
        sizes['interval_count'] = INTERVAL_COUNTS[turn]

    # length of model output vector
    sizes['out'] = NUM_OUT[name]

    dump(sizes, INPUT_DIR + 'sizes/{}.pkl'.format(name))


def convert_x_to_numpy(x, idx):
    """
    Converts dictionary of dataframes to dictionary of numpy arrays.
    :param x: dictionary of input dataframes.
    :param idx: pandas index for error checking indices.
    :return: dictionary of numpy arrays.
    """
    for k, v in x.items():
        assert np.all(v.index == idx)
        x[k] = v.to_numpy(dtype='float32')

    return x


def save_small(d, name):
    # randomly select indices
    v = np.arange(np.shape(d['y'])[0])
    np.random.shuffle(v)
    idx_small = v[:N_SMALL]

    # outcome
    small = dict()
    small['y'] = d['y'][idx_small]

    # inputs
    small['x'] = {k: v[idx_small, :] for k, v in d['x'].items()}

    # save
    dump(small, INPUT_DIR + 'small/{}.gz'.format(name))


# save featnames and sizes
def save_files(d, part, name):
    # featnames and sizes
    if part == VALIDATION:
        save_featnames(d['x'], name)
        save_sizes(d['x'], name)

    # pandas index
    idx = d['y'].index

    # input features
    d['x'] = convert_x_to_numpy(d['x'], idx)

    # convert outcome to numpy
    d['y'] = d['y'].to_numpy()

    # save data
    dump(d, INPUT_DIR + '{}/{}.gz'.format(part, name))

    # save index
    dump(idx, INDEX_DIR + '{}/{}.gz'.format(part, name))

    # save subset
    if part == TRAIN_MODELS:
        save_small(d, name)
