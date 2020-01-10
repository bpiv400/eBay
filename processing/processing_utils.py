import sys, os, argparse, h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
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


def get_arrival(lstg_start, thread_start):
    # intervals until thread
    thread_periods = (thread_start - lstg_start) // INTERVAL['arrival']

    # error checking
    assert thread_periods.max() < INTERVAL_COUNTS['arrival']

    # count of arrivals by interval
    y = thread_periods.rename('period').to_frame().assign(
        count=1).groupby(['lstg', 'period']).sum().squeeze()

    return y


def get_first_index(idx_target, idx_master):
    s = pd.Series(range(len(idx_target)), index=idx_target, name='first')
    s = s.groupby(s.index.names[:-1]).first()
    s = s.reindex(idx_master, fill_value=-1)
    return s


def periods_to_string(idx_target, idx_master):
    # create series with (string) period as values
    s = pd.DataFrame(index=idx_target).reset_index('period').squeeze().astype(str)

    # collapse to '/'-connected strings
    s = s.groupby(s.index.names).agg(lambda x: '/'.join(x.tolist()))

    # expand to master index
    s = s.reindex(index=idx_master, fill_value='')

    return s.astype('S')


def reshape_indices(idx_target, idx_master):
    # call helper functions
    periods = periods_to_string(idx_target, idx_master)
    indices = get_first_index(idx_target, idx_master)

    # error checking
    assert all(periods[indices == -1] == ''.encode('ascii'))
    assert all(indices[periods == ''.encode('ascii')] == -1)

    return periods, indices


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


def get_x_thread(part, idx):
    # thread features
    df = load_file(part, 'x_thread')

    # byr_hist as a decimal
    df.loc[:, 'byr_hist'] = df.byr_hist.astype('float32') / 10

    # reindex to create x_thread
    x_thread = pd.DataFrame(index=idx).join(df)

    # add turn indicators
    x_thread = add_turn_indicators(x_thread)

    return x_thread.astype('float32')


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
        offer = add_turn_indicators(offer)

        # float32s
        x['offer%d' % i] = offer.astype('float32')

    return x


def get_idx_x(part, idx):
    # create series with lstgs in x as index
    lstgs_x = load_file(part, 'lookup').index
    s = pd.Series(range(len(lstgs_x)), index=lstgs_x, name='idx_x')

    # merge target indices with master indices
    df = pd.DataFrame(index=idx)
    idx_x = df.join(s, on='lstg', sort=False).squeeze()

    # error checking
    assert all(lstgs_x[idx_x.values] == idx_x.index.get_level_values(level='lstg'))

    return idx_x


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


def get_featnames(d, name):
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

    # for discriminator models
    if name == 'listings':
        featnames['x']['arrival'] = ['arrival{}'.format(i) \
            for i in range(INTERVAL_COUNTS[ARRIVAL_PREFIX])]

    return featnames


def save_sizes(d, featnames, name):
    '''
    Creates dictionary of input sizes.
    :param d: dictionary of inputs.
    :param featnames: dictionary of featnames.
    :param name: string name of model.
    '''
    sizes = {}

    # counts of examples (and labels)
    if 'periods' in d:
        sizes['N_examples'] = len(d['periods'])
        sizes['N_labels'] = d['periods'].sum()
    else:
        sizes['N_labels'] = len(d['y'])

    # count components of x
    sizes['x'] = {}
    for k, v in featnames['x'].items():
        sizes['x'][k] = len(v)

    # arrival and delay models
    if 'x_time' in featnames:
        role = name.split('_')[-1]
        sizes['interval'] = INTERVAL[role]
        sizes['interval_count'] = INTERVAL_COUNTS[role]
        if role == BYR_PREFIX:
            sizes['interval_count_7'] = INTERVAL_COUNTS[BYR_PREFIX + '_7']

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
    # for error checking
    if 'periods' in d:
        master_idx = d['periods'].index
    else:
        master_idx = d['y'].index

    # error checking
    for k in d.keys():
        if 'periods' in k or 'idx' in k or k in ['y', 'seconds', 'x_thread', 'remaining']:
            assert np.all(d[k].index == master_idx)

    # convert dataframes to numpy
    for k, v in d.items():
        if not isinstance(v, dict):
            d[k] = v.to_numpy()

    # loop through x_offer, error checking and convert to numpy
    if 'x_offer' in d:
        x_offer = d.pop('x_offer')
        for k, v in x_offer.items():
            assert np.all(v.index == master_idx)
            d[k] = v.to_numpy()

    return d

# save featnames and sizes
def save_files(d, part, name):
    # featnames and sizes
    if part == 'train_models':
        # featnames
        featnames = get_featnames(d, name)
        dump(featnames, INPUT_DIR + 'featnames/{}.pkl'.format(name))

        # sizes
        save_sizes(d, featnames, name)

    # create dictionary of numpy arrays
    d = convert_to_numpy(d)

    # for recurrent models, save periods separately
    if 'periods' in d:
        periods = d.pop('periods')
        dump(periods, INPUT_DIR + '{}/{}.gz'.format(part, name))

    # save hdf5 file
    with h5py.File(HDF5_DIR + '{}/{}.hdf5'.format(part, name), 'w') as f:
        for k, v in d.items():
            f.create_dataset(k, data=v)