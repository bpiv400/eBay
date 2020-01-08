import sys, os, argparse
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from processing.processing_consts import *
from constants import *
from featnames import *


# split into chunks and save
def chunk(group, L, T, O):
    S = L[group].reset_index().set_index(group).squeeze()
    counts = S.groupby(S.index.name).count()
    groups = []
    total = 0
    num = 1
    for i in range(len(counts)):
        groups.append(counts.index[i])
        total += counts.iloc[i]
        if (i == len(counts)-1) or (total >= CUTOFF):
            # find correspinding listings
            idx = S.loc[groups]
            # create chunks
            L_i = L.reindex(index=idx)
            T_i = T.reindex(index=idx, level='lstg')
            O_i = O.reindex(index=idx, level='lstg')
            # save
            chunk = {'listings': L_i, 'threads': T_i, 'offers': O_i}
            path = CHUNKS_DIR + '%s%d.gz' % (group, num)
            dump(chunk, path)
            # reinitialize
            groups = []
            total = 0
            # increment
            num += 1


def extract_day_feats(clock):
    """
    Returns dataframe with US holiday and day-of-week indicators
    :param clock: pandas series of timestamps
    :return: dataframe with holiday and day of week indicators
    """
    df = pd.DataFrame(index=clock.index)
    df['holiday'] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df['dow' + str(i)] = clock.dt.dayofweek == i

    return df


def extract_clock_feats(clock):
    '''
    Creates clock features from timestamps.
    :param clock: pandas series of timestamps.
    :return: pandas dataframe of holiday and day of week indicators, and minute of day.
    '''
    df = extract_day_feats(clock)

    # add in seconds of day
    seconds_since_midnight = clock.dt.hour * 60 + clock.dt.minute + clock.dt.second
    df['second_of_day'] = (seconds_since_midnight / (24 * 3600)).astype('float32')

    assert all(df.columns == CLOCK_FEATS)

    return df


# loads processed chunk files
def load_frames(name):
    '''
    name: one of 'events', 'tf_lstg', 'tf_slr'.
    '''
    # path to file number x
    path = lambda num: FEATS_DIR + '{}_{}.gz'.format(num, name)
    # loop and append
    output = []
    n = len([f for f in os.listdir(FEATS_DIR) if name in f])
    for i in range(1,n+1):
        output.append(load(path(i)))
    output = pd.concat(output).sort_index()
    return output


# function to load file
def load_file(part, x):
    return load(PARTS_DIR + '{}/{}.gz'.format(part, x))


# returns partition indices and path to file function
def get_partition(part):
    '''
    part: one of 'train_models', 'train_rl', 'test'
    '''
    partitions = load(PARTS_DIR + 'partitions.pkl')
    idx = partitions[part]
    path = lambda name: PARTS_DIR + '%s/%s.gz' % (part, name)
    return idx, path


def input_partition():
    """
    Parses command line input for partition name.
    :return part: string partition name.
    """
    parser = argparse.ArgumentParser()

    # partition
    parser.add_argument('--part', required=True, type=str, help='partition name')
    part = parser.parse_args().part

    # error checking
    if part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))

    return part


def add_turn_indicators(df):
    '''
    Appends turn indicator variables to offer matrix
    :param df: dataframe with index ['lstg', 'thread', 'index'].
    :return: dataframe with turn indicators appended
    '''
    indices = np.unique(df.index.get_level_values('index'))
    for i in range(len(indices)-1):
        ind = indices[i]
        featname = 't%d' % ((ind+1) // 2)
        df[featname] = df.index.isin([ind], level='index')
    return df


# deletes irrelevant feats and sets unseen feats to 0
def clean_offer(offer, i, outcome, role, dtypes):
    # set features to 0 if i exceeds index
    if i > 1:
        future = i > offer.index.get_level_values(level='index')
        offer.loc[future, dtypes == 'bool'] = False
        offer.loc[future, dtypes != 'bool'] = 0
    # for current turn, set feats to 0
    curr = i == offer.index.get_level_values(level='index')
    if outcome == 'delay':
        offer.loc[curr, dtypes == 'bool'] = False
        offer.loc[curr, dtypes != 'bool'] = 0
    else:
        offer.loc[curr, 'msg'] = False
        if outcome == 'con':
            offer.loc[curr, ['con', 'norm']] = 0
            offer.loc[curr, ['split', 'auto', 'exp', 'reject']] = False
    # if turn 1, drop days and delay
    if i == 1:
        offer = offer.drop(['days', 'delay'], axis=1)
    # if buyer turn or last turn, drop auto, exp, reject
    if (i in IDX['byr']) or (i == max(IDX[role])):
        offer = offer.drop(['auto', 'exp', 'reject'], axis=1)
    # on last turn, drop msg (and concession features)
    if i == max(IDX[role]):
        offer = offer.drop('msg', axis=1)
        if outcome == 'con':
            offer = offer.drop(['con', 'norm', 'split'], axis=1)
    return offer


def get_x_thread(part, idx):
    # thread features
    df = load_file(part, 'x_thread')

    # byr_hist as a decimal
    df.loc[:, 'byr_hist'] = df.byr_hist.astype('float32') / 10

    # reindex to create x_thread
    x_thread = pd.DataFrame(index=idx).join(df)

    # add turn indicators
    x_thread = add_turn_indicators(x_thread)

    return x_thread


def get_x_offer(part, idx, outcome=None, role=None):
    # offer features
    df = load_file(part, 'x_offer')

    # initialize dictionary of offer features
    x = {}

    # dataframe of offer features for relevant threads
    threads = idx.droplevel(level='index').unique()
    df = pd.DataFrame(index=threads).join(df)
    dtypes = df.dtypes

    # turn features
    for i in range(1, max(IDX[role])+1):
        # offer features at turn i
        offer = df.xs(i, level='index').reindex(index=idx)

        # clean
        offer = clean_offer(offer, i, outcome, role, dtypes)

        # add turn number to featname
        offer = offer.rename(lambda x: x + '_%d' % i, axis=1)

        # add turn indicators
        x['offer%d' % i] = add_turn_indicators(offer)

    return x


def get_idx_x(part, idx):
    lstgs_x = load_file(part, 'lookup').index.to_numpy()
    lstgs_y = idx.get_level_values(level='lstg').to_numpy()
    return [np.nonzero(lstgs_x == n)[0][0] for n in lstgs_y]


def get_tf(tf, start, periods, role):
    # add period to tf_arrival
    tf = tf.join(start.rename('start'))
    tf['period'] = (tf.clock - tf.start) // INTERVAL[role]
    tf = tf.drop(['clock', 'start'], axis=1)

    # increment period by 1; time feats are up to t-1
    tf['period'] += 1

    # drop periods beyond censoring threshold
    tf = tf.join(periods.rename('periods'))
    tf = tf[tf.period < tf.periods]
    tf = tf.drop('periods', axis=1)

    # sum count features by period and return
    return tf.groupby(list(tf.index.names) + ['period']).sum()


def save_featnames(d, name):
    '''
    Creates dictionary of input feature names.
    :param d: dictionary with dataframes.
    :param name: string name of model.
    '''

    # initialize with components of x
    featnames = {}
    featnames['x'] = load(INPUT_DIR + 'featnames/x_lstg.pkl')
    if 'x_thread' in d:
        featnames['x']['lstg'] += list(d['x_thread'].columns)

    if 'x_offer' in d:
        for k, v in d['x_offer'].items():
            featnames['x'][k] = list(v.columns)

    # for arrival and delay models
    if 'periods' in d:
        featnames['x_time'] = \
            CLOCK_FEATS + TIME_FEATS + [DURATION]

        # check that time feats match names
        assert list(d['tf'].columns) == TIME_FEATS

        # for delay models
        if 'remaining' in d:
            featnames['x_time'] += [INT_REMAINING]
            assert INT_REMAINING == 'remaining'

    dump(featnames, INPUT_DIR + 'featnames/{}.pkl'.format(name))


def save_sizes(featnames, name):
    '''
    Creates dictionary of input sizes.
    :param featnames: dictionary of featnames.
    :param name: string name of model.
    '''
    sizes = {}

    # count components of x
    sizes['x'] = {}
    for k, v in featnames['x'].items():
        sizes['x'][k] = len(v)

    # arrival and delay models
    if 'x_time' in featnames:
        role = name.split('_')[-1]
        sizes['interval'] = INTERVAL[role]
        sizes['interval_count'] = INTERVAL_COUNTS[role]
        if role == 'byr':
            sizes['interval_count_7'] = INTERVAL_COUNTS['byr_7']

        sizes['x_time'] = len(featnames['x_time'])

    # output size is based on model
    if name == 'hist':
        sizes['out'] = HIST_QUANTILES
    elif 'con' in name:
        sizes['out'] = 101
    else:
        sizes['out'] = 1

    dump(sizes, INPUT_DIR + 'sizes/{}.pkl'.format(name))


def convert_to_numpy(d):
    '''
    Converts dictionary of dataframes to dictionary of numpy arrays.
    :param d: dictionary with (dictionaries of) dataframes.
    :return: dictionary with numpy arrays (and dictionaries of dataframes).
    '''

    # loop through x_offer, convert to numpy
    if 'x_offer' in d:
        for k, v in d['x_offer'].items():
            d['x_offer'][k] = v.to_numpy(dtype='float32')

    # lists for recurrent components
    if 'periods' in d:
        master_idx = d['periods'].index
        for k in ['y', 'tf']:
            if k == 'y':
                s = d['y']
            else:
                s = pd.Series(
                        d['tf'].values.astype('float32').tolist(), 
                        index=d['tf'].index)
            indices = s.reset_index(-1).index
            d[k] = []
            for idx in master_idx:
                if idx in indices:
                    d[k].append(s.xs(idx).to_dict())
                else:
                    d[k].append({})

    # for error checking
    else:
        master_idx = d['y'].index

    # convert remaining components to numpy directly
    for k, v in d.items():
        if not isinstance(v, (list, dict)):
            assert np.all(v.index == master_idx)
            d[k] = v.to_numpy(dtype='float32')

    # x_idx is a list
    d['idx_x'] = np.array(d['idx_x'])

    return d


# save featnames and sizes
def save_files(d, part, name):
    # featnames and sizes
    if part == 'train_models':
        save_featnames(d, name)
        save_sizes(featnames, name)

    # create dictionary of numpy arrays
    d = convert_to_numpy(d)

    # save as dataset
    dump(d, INPUT_DIR + '{}/{}.gz'.format(part, name))