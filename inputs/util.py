from collections import OrderedDict
import numpy as np
import pandas as pd
from utils import unpickle, topickle
from inputs.const import NUM_OUT
from constants import INPUT_DIR, INDEX_DIR, VALIDATION, \
    IDX, DISCRIM_MODEL, POLICY_BYR, POLICY_MODELS, BYR_DROP
from featnames import CLOCK_FEATS, OUTCOME_FEATS, SPLIT, MSG, AUTO, LSTG, X_LSTG, \
    EXP, REJECT, DAYS, DELAY, TIME_FEATS, THREAD_COUNT, BYR, INDEX, SLR, THREAD, \
    META, LEAF


def add_turn_indicators(df):
    """
    Appends turn indicator variables to offer matrix
    :param df: dataframe with index ['lstg', 'thread', 'index'].
    :return: dataframe with turn indicators appended
    """
    indices = np.sort(np.unique(df.index.get_level_values(INDEX)))
    for i in range(len(indices) - 1):
        ind = indices[i]
        df['t{}'.format(ind)] = df.index.isin([ind], level=INDEX)
    return df


def get_x_thread(threads, idx, turn_indicators=False):
    # initialize x_thread as copy
    x_thread = threads.copy()
    # thread count, including current thread
    x_thread[THREAD_COUNT] = x_thread.index.get_level_values(level=THREAD)
    # reindex to create x_thread
    x_thread = pd.DataFrame(index=idx).join(x_thread)
    x_thread.index = x_thread.index.reorder_levels(idx.names)
    if turn_indicators:
        x_thread = add_turn_indicators(x_thread)
    return x_thread.astype('float32')


def get_arrival_times(clock=None, lstg_start=None, lstg_end=None, append_last=False):
    # thread 0: start of listing
    s = lstg_start.to_frame().assign(thread=0).set_index(
        'thread', append=True).squeeze()

    # threads 1 to N: real threads
    thread_start = clock.xs(1, level='index')
    threads = thread_start.reset_index('thread').drop(
        'clock', axis=1).squeeze().groupby('lstg').max().reindex(
        index=lstg_start.index, fill_value=0)

    # thread N+1: end of lstg
    if append_last:
        s1 = lstg_end.to_frame().assign(
            thread=threads + 1).set_index(
            'thread', append=True).squeeze()
        s = pd.concat([s, s1], axis=0)

    # concatenate and sort into single series
    arrivals = pd.concat([s, thread_start], axis=0).sort_index()

    # thread to int
    idx = arrivals.index
    arrivals.index.set_levels(idx.levels[-1].astype('int16'),
                              level=-1, inplace=True)

    return arrivals.rename('arrival')


def assert_zero(offer, cols, k):
    for c in cols:
        if offer[c].max() != 0 or offer[c].min() != 0:
            raise RuntimeError('Non-zero entry in {}: {}'.format(k, c))


def check_zero(x, byr_exp=True):
    keys = [k for k in x.keys() if k.startswith('offer')]
    byr_cols = [AUTO, REJECT]
    if byr_exp:
        byr_cols += [EXP]
    for k in keys:
        if k in x:
            i = int(k[-1])
            if i == 1:
                assert_zero(x[k], [DAYS, DELAY], k)
            if i % 2 == 1:
                assert_zero(x[k], byr_cols, k)
            if i == 7:
                assert_zero(x[k], [SPLIT, MSG], k)


def get_ind_x(lstgs=None, idx=None):
    """
    For each lstg in idx, finds corresponding index in lstgs.
    :param lstgs: index of lookup.
    :param idx: index of outcome.
    :return: array of indices in lstgs.
    """
    idx = idx.get_level_values(level=LSTG)  # restrict to lstg id
    idx_x = np.searchsorted(lstgs, idx)
    return idx_x


def save_featnames_and_sizes(x=None, m=None):
    """
    Creates dictionary of input feature names and sizes.
    :param dict x: input dataframes.
    :param str m: name of model.
    """
    # initialize dictionaries from listing feature names
    featnames = unpickle(INPUT_DIR + 'featnames/{}.pkl'.format(X_LSTG))
    sizes = dict()
    sizes['x'] = OrderedDict()
    for k, v in featnames.items():
        sizes['x'][k] = len(v)

    if m in POLICY_MODELS:
        for k in [SLR, META, LEAF]:
            del featnames[k]
            del sizes['x'][k]

    if m == POLICY_BYR:
        featnames[LSTG] = [k for k in featnames[LSTG] if k not in BYR_DROP]
        sizes['x'][LSTG] = len(featnames[LSTG])

    # add thread features to end of lstg grouping
    if x is not None:
        featnames[LSTG] += list(x[THREAD].columns)
        sizes['x'][LSTG] += len(x[THREAD].columns)

        # for offer models
        if 'offer1' in x:
            if m == DISCRIM_MODEL:
                for i in range(1, 8):
                    k = 'offer{}'.format(i)
                    featnames[k] = list(x[k].columns)
                    sizes['x'][k] = len(featnames[k])
            else:
                # buyer models do not have time feats
                if m == POLICY_BYR or m[-1] in [str(i) for i in IDX[BYR]]:
                    feats = CLOCK_FEATS + OUTCOME_FEATS
                else:
                    feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS

                # check that all offer groupings have same organization
                for k in x.keys():
                    if 'offer' in k:
                        assert list(x[k].columns) == feats
                        sizes['x'][k] = len(feats)  # put length in sizes

                # one vector of featnames for offer groupings
                featnames['offer'] = feats

    # length of model output vector
    sizes['out'] = NUM_OUT[m]

    # save
    topickle(featnames, INPUT_DIR + 'featnames/{}.pkl'.format(m))
    topickle(sizes, INPUT_DIR + 'sizes/{}.pkl'.format(m))


def convert_x_to_numpy(x, idx):
    """
    Converts dictionary of dataframes to dictionary of numpy arrays.
    :param dict x: contains input dataframes.
    :param pd.(Multi)Index idx: for error checking indices.
    :return: dict of numpy arrays.
    """
    for k, v in x.items():
        assert np.all(v.index == idx)
        x[k] = v.to_numpy(dtype='float32')


def save_files(d, part, name):
    # featnames and sizes
    if part == VALIDATION:
        save_featnames_and_sizes(x=None if 'x' not in d else d['x'],
                                 m=name)

    # pandas index
    idx = d['y'].index

    # input features
    if 'x' in d:
        convert_x_to_numpy(d['x'], idx)

    # convert outcome to numpy
    d['y'] = d['y'].to_numpy()

    # save data
    topickle(d, INPUT_DIR + '{}/{}.pkl'.format(part, name))

    # save index
    topickle(idx, INDEX_DIR + '{}/{}.pkl'.format(part, name))
