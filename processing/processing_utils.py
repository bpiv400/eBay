import argparse
import numpy as np
import pandas as pd
from compress_pickle import load, dump
from utils import slr_norm, byr_norm, extract_clock_feats, get_months_since_lstg
from processing.processing_consts import *
from constants import *
from featnames import *


# function to load file from partitions directory
def load_file(part, x):
    return load(PARTS_DIR + '{}/{}.gz'.format(part, x))


def extract_day_feats(seconds):
    """
    Creates clock features from timestamps.
    :param seconds: seconds since START.
    :return: tuple of time_of_day sine transform and afternoon indicator.
    """
    clock = pd.to_datetime(seconds, unit='s', origin=START)
    df = pd.DataFrame(index=clock.index)
    df[HOLIDAY] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df[DOW_PREFIX + str(i)] = clock.dt.dayofweek == i
    return df


def collect_date_clock_feats(seconds):
    """
    Combines date and clock features.
    :param seconds: seconds since START.
    :return: dataframe of date and clock features.
    """
    df = extract_day_feats(seconds)
    df[TIME_OF_DAY], df[AFTERNOON] = extract_clock_feats(seconds)
    assert list(df.columns) == CLOCK_FEATS
    return df


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


def get_days_delay(clock):
    """
    Calculates time between successive offers.
    :param clock: dataframe with index ['lstg', 'thread'], 
        turn numbers as columns, and seconds since START as values
    :return days: fractional number of days between offers.
    :return delay: time between offers as share of MAX_DELAY.
    """
    # initialize output dataframes in wide format
    days = pd.DataFrame(0., index=clock.index, columns=clock.columns)
    delay = pd.DataFrame(0., index=clock.index, columns=clock.columns)

    # for turn 1, days and delay are 0
    for i in range(2, 8):
        days[i] = clock[i] - clock[i - 1]
        if i in [2, 4, 6, 7]:  # byr has 2 days for last turn
            delay[i] = days[i] / MAX_DELAY[SLR_PREFIX]
        elif i in [3, 5]:  # ignore byr arrival and last turn
            delay[i] = days[i] / MAX_DELAY[BYR_PREFIX]
    # no delay larger than 1
    assert delay.max().max() <= 1

    # reshape from wide to long
    days = days.rename_axis('index', axis=1).stack() / DAY
    delay = delay.rename_axis('index', axis=1).stack()

    return days, delay


def round_con(con):
    """
    Round concession to nearest percentage point.
    :param con: pandas series of unrounded concessions.
    :return: pandas series of rounded concessions.
    """
    rounded = np.round(con, decimals=2)
    rounded.loc[(rounded == 1) & (con < 1)] = 0.99
    rounded.loc[(rounded == 0) & (con > 0)] = 0.01
    return rounded


def get_con(offers, start_price):
    # compute concessions
    con = pd.DataFrame(index=offers.index)
    con[1] = offers[1] / start_price
    con[2] = (offers[2] - start_price) / (offers[1] - start_price)
    for i in range(3, 8):
        con[i] = (offers[i] - offers[i - 2]) / (offers[i - 1] - offers[i - 2])

    # stack into series
    con = con.rename_axis('index', axis=1).stack()

    # first buyer concession should be greater than 0
    assert con.loc[con.index.isin([1], level='index')].min() > 0

    # round concessions
    rounded = round_con(con)

    return rounded


def get_norm(con):
    """
    Calculate normalized concession from rounded concessions.
    :param con: pandas series of rounded concessions.
    :return: pandas series of normalized concessions.
    """
    df = con.unstack()
    norm = pd.DataFrame(index=df.index, columns=df.columns)
    norm[1] = df[1]
    norm[2] = df[2] * (1 - norm[1])
    for i in range(3, 8):
        if i in IDX[BYR_PREFIX]:
            norm[i] = byr_norm(con=df[i],
                               prev_byr_norm=norm[i - 2],
                               prev_slr_norm=norm[i - 1])
        elif i in IDX[SLR_PREFIX]:
            norm[i] = slr_norm(con=df[i],
                               prev_byr_norm=norm[i - 1],
                               prev_slr_norm=norm[i - 2])
    return norm.rename_axis('index', axis=1).stack().astype('float64')


def add_turn_indicators(df):
    """
    Appends turn indicator variables to offer matrix
    :param df: dataframe with index ['lstg', 'thread', 'index'].
    :return: dataframe with turn indicators appended
    """
    indices = np.sort(np.unique(df.index.get_level_values('index')))
    for i in range(len(indices) - 1):
        ind = indices[i]
        df['t{}'.format(ind)] = df.index.isin([ind], level='index')
    return df


def get_x_thread(threads, idx):
    x_thread = threads.copy()

    # byr_hist as a decimal
    x_thread.loc[:, BYR_HIST] = x_thread.byr_hist.astype('float32') / 10

    # reindex to create x_thread
    x_thread = pd.DataFrame(index=idx).join(x_thread)

    # add turn indicators
    if 'index' in idx.names:
        x_thread = add_turn_indicators(x_thread)

    return x_thread.astype('float32')


# sets unseen feats to 0
def set_zero_feats(offer, i, outcome):
    # turn number
    turn = offer.index.get_level_values(level='index')

    # all features are zero for future turns
    if i > 1:
        offer.loc[i > turn, :] = 0.0

    # for current turn, set feats to 0
    curr = i == turn
    if outcome == DELAY:
        offer.loc[curr, :] = 0.0
    else:
        offer.loc[curr, MSG] = 0.0
        if outcome == CON:
            offer.loc[curr, [CON, NORM, SPLIT, AUTO, EXP, REJECT]] = 0.0

    return offer


def get_x_offer(offers, idx, outcome=None, role=None):
    # initialize dictionary of offer features
    x_offer = {}

    # for threads set role to byr
    if outcome is None and role is None:
        role = BYR_PREFIX

    # dataframe of offer features for relevant threads
    if 'index' in idx.names:
        threads = idx.droplevel(level='index').unique()
    else:
        threads = idx
    offers = pd.DataFrame(index=threads).join(offers)

    # last turn to include
    last = max(IDX[role])
    if outcome == DELAY:
        last -= 1
    if (outcome == MSG) & (role == BYR_PREFIX):
        last -= 2

    # turn features
    for i in range(1, last + 1):
        # offer features at turn i
        offer = offers.xs(i, level='index').reindex(
            index=idx, fill_value=0).astype('float32')

        # set unseen feats to 0 and add turn indicators
        if outcome is not None:
            offer = set_zero_feats(offer, i, outcome)
            offer = add_turn_indicators(offer)

        # drop time feats from buyer models
            if role == BYR_PREFIX:
                offer = offer.drop(TIME_FEATS, axis=1)

        # set censored time feats to zero
        else:
            if i > 1:
                censored = (offer[EXP] == 1) & (offer[DELAY] < 1)
                offer.loc[censored, TIME_FEATS] = 0.0

        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')

    return x_offer


def init_x(part, idx, drop_slr=False):
    x = load_file(part, 'x_lstg')
    x = {k: v.reindex(index=idx, level='lstg').astype('float32') for k, v in x.items()}
    if drop_slr:
        del x['slr']
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
    y //= INTERVAL[ARRIVAL_PREFIX]

    # replace censored interarrival times negative count of censored buckets
    y.loc[censored] -= INTERVAL_COUNTS[ARRIVAL_PREFIX]

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
        if BYR_PREFIX in name:
            feats = CLOCK_FEATS + OUTCOME_FEATS + TURN_FEATS[name]
        else:
            feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS + TURN_FEATS[name]

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
    role = name.split('_')[-1]
    if ('arrival' in name) or ('delay' in name):
        sizes['interval'] = INTERVAL[role]
        sizes['interval_count'] = INTERVAL_COUNTS[role]
        if role == BYR_PREFIX:
            sizes['interval_count_7'] = INTERVAL_COUNTS[BYR_PREFIX + '_7']

    # length of model output vector
    if 'arrival' in name:
        sizes['out'] = INTERVAL_COUNTS[ARRIVAL_PREFIX] + 1
    elif name == 'hist':
        sizes['out'] = HIST_QUANTILES
    elif DELAY in name:
        sizes['out'] = INTERVAL_COUNTS[role] + 1
    elif CON in name:
        sizes['out'] = CON_MULTIPLIER + 1
    else:
        sizes['out'] = 1

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
        x[k] = v.to_numpy()

    return x


def save_small(d, name):
    # randomly select indices
    v = np.arange(np.shape(d['y'])[0])
    np.random.shuffle(v)
    idx_small = v[:N_SMALL]

    # outcome
    small = dict()
    small['y'] = d['y'][idx_small]

    # baserates
    small['baserates'] = d['baserates'][idx_small]

    # inputs
    small['x'] = {k: v[idx_small, :] for k, v in d['x'].items()}

    # save
    dump(small, INPUT_DIR + 'small/{}.gz'.format(name))


def get_baserates(y, name):
    if MSG in name:
        # probability of y == 1
        if 'index' in y.index.names:
            p = y.groupby('index').mean().to_dict()
        else:
            p = y.mean()
        return p

    if CON in name:
        # probability of each concession value by turn
        p = dict()
        if 'index' in y.index.names:
            for turn in y.index.unique(level='index'):
                p[turn] = np.zeros(CON_MULTIPLIER + 1, dtype='float32')
                s = y.xs(turn, level='index')
                den = len(s) + CON_MULTIPLIER + 1
                for i in range(CON_MULTIPLIER + 1):
                    p[turn][i] = (1 + (s == i).sum()) / den
        else:
            p = np.zeros(CON_MULTIPLIER + 1, dtype='float32')
            den = len(y) + CON_MULTIPLIER + 1
            for i in range(CON_MULTIPLIER + 1):
                p[i] = (1 + (y == i).sum()) / den
        return p

    raise NotImplementedError()


# save featnames and sizes
def save_files(d, part, name):
    # featnames and sizes
    if part == 'test_rl':
        save_featnames(d['x'], name)
        save_sizes(d['x'], name)

    # baserates
    d['p'] = get_baserates(d['y'], name)

    # turn index for baserates
    if 'index' in d['y'].index.names:
        d['p_idx'] = d['y'].index.get_level_values(level='index').to_numpy()

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
