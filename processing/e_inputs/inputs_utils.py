import numpy as np
import pandas as pd
from compress_pickle import dump
from processing.processing_utils import load_file
from utils import get_remaining
from processing.processing_consts import *
from constants import *
from featnames import *


def calculate_remaining(part, idx, turn=None):
    # load timestamps
    lstg_start = load_file(part, 'lookup').start_time.reindex(
        index=idx, level='lstg')
    delay_start = load_file(part, 'clock').groupby(
        ['lstg', 'thread']).shift().dropna().astype('int64')

    if turn is None:
        remaining = pd.Series(np.nan, index=idx)
        for turn in idx.unique(level='index'):
            turn_start = delay_start.xs(turn, level='index').reindex(index=idx)
            mask = idx.get_level_values(level='index') == turn
            remaining.loc[mask] = get_remaining(
                lstg_start, turn_start, MAX_DELAY[turn])
    else:
        assert 'index' not in idx.names
        turn_start = delay_start.xs(turn, level='index').reindex(index=idx)
        remaining = get_remaining(lstg_start, turn_start, MAX_DELAY[turn])

    # error checking
    assert np.all(remaining > 0) and np.all(remaining <= 1)

    return remaining


def get_x_thread(threads, idx):
    # initialize x_thread as copy
    x_thread = threads.copy()

    # byr_hist as a decimal
    x_thread.loc[:, BYR_HIST] = x_thread.byr_hist.astype('float32') / 10

    # reindex to create x_thread
    x_thread = pd.DataFrame(index=idx).join(x_thread)

    return x_thread.astype('float32')


def init_x(part, idx):
    x = load_file(part, 'x_lstg')
    x = {k: v.reindex(index=idx, level='lstg').astype('float32') for k, v in x.items()}
    return x


def get_interarrival_period(clock):
    # calculate interarrival times in seconds
    df = clock.unstack()
    diff = pd.DataFrame(0.0, index=df.index, columns=df.columns[1:])
    for i in diff.columns:
        diff[i] = df[i] - df[i - 1]

    # restack
    diff = diff.rename_axis(clock.index.names[-1], axis=1).stack()

    # original datatype
    diff = diff.astype(clock.dtype)

    # indicator for whether observation is last in lstg
    thread = pd.Series(diff.index.get_level_values(level='thread'),
                       index=diff.index)
    last_thread = thread.groupby('lstg').max().reindex(
        index=thread.index, level='lstg')
    censored = thread == last_thread

    # drop interarrivals after BINs
    diff = diff[diff > 0]
    y = diff[diff.index.get_level_values(level='thread') > 1]
    censored = censored.reindex(index=y.index)

    # convert y to periods
    y //= INTERVAL[1]

    # replace censored interarrival times negative count of censored buckets
    y.loc[censored] -= INTERVAL_COUNTS[1]

    return y, diff


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


def get_y_con(df):
    # drop zero delay and expired offers
    mask = ~df[AUTO] & ~df[EXP]
    # concession is an int from 0 to 100
    return (df.loc[mask, CON] * CON_MULTIPLIER).astype('int8')


def get_y_msg(df, turn):
    # for buyers, drop accepts and rejects
    if turn in IDX[BYR_PREFIX]:
        mask = (df[CON] > 0) & (df[CON] < 1)
    # for sellers, drop accepts, expires, and auto responses
    else:
        mask = ~df[EXP] & ~df[AUTO] & (df[CON] < 1)
    return df.loc[mask, MSG]


def assert_zero(offer, cols):
    for c in cols:
        assert offer[c].max() == 0
        assert offer[c].min() == 0


def check_zero(x):
    keys = [k for k in x.keys() if k.startswith('offer')]
    for k in keys:
        if k in x:
            i = int(k[-1])
            if i == 1:
                assert_zero(x[k], [DAYS, DELAY])
            if i % 2 == 1:
                assert_zero(x[k], [AUTO, EXP, REJECT])
            if i == 7:
                assert_zero(x[k], [SPLIT, MSG])


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
    if part == 'test_rl':
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
    if part == 'train_models':
        save_small(d, name)


def save_discrim_files(part, name, x_obs, x_sim):
    # featnames and sizes
    if part == 'test_rl':
        save_sizes(x_obs, name)

    # indices
    idx_obs = x_obs['lstg'].index
    idx_sim = x_sim['lstg'].index

    # create dictionary of numpy arrays
    x_obs = convert_x_to_numpy(x_obs, idx_obs)
    x_sim = convert_x_to_numpy(x_sim, idx_sim)

    # y=1 for real data
    y_obs = np.ones(len(idx_obs), dtype=bool)
    y_sim = np.zeros(len(idx_sim), dtype=bool)
    d = {'y': np.concatenate((y_obs, y_sim), axis=0)}

    # join input variables
    assert x_obs.keys() == x_sim.keys()
    d['x'] = {k: np.concatenate((x_obs[k], x_sim[k]), axis=0) for k in x_obs.keys()}

    # save inputs
    dump(d, INPUT_DIR + '{}/{}.gz'.format(part, name))

    # save small
    if part == 'train_rl':
        save_small(d, name)
