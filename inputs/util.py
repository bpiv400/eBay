import numpy as np
import pandas as pd
from compress_pickle import dump
from processing.util import extract_day_feats
from utils import get_remaining, load_file, init_x
from inputs.const import NUM_OUT, N_SMALL, INTERVAL, \
    INTERVAL_COUNTS, DELTA_MONTH, DELTA_ACTION, C_ACTION
from constants import INPUT_DIR, INDEX_DIR, VALIDATION, TRAIN_MODELS, \
    TRAIN_RL, IDX, BYR_PREFIX, DELAY_MODELS, SLR_PREFIX, DAY, MONTH, \
    INIT_VALUE_MODELS, ARRIVAL_PREFIX, MAX_DELAY, NO_ARRIVAL_CUTOFF
from featnames import CLOCK_FEATS, OUTCOME_FEATS, BYR_HIST, \
    CON, NORM, SPLIT, MSG, AUTO, EXP, REJECT, DAYS, DELAY, TIME_FEATS, \
    THREAD_COUNT, MONTHLY_DISCOUNT, ACTION_DISCOUNT, ACTION_COST, \
    NO_ARRIVAL, START_TIME, INT_REMAINING, MONTHS_SINCE_LSTG


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


def get_x_thread(threads, idx, turn_indicators=False):
    # initialize x_thread as copy
    x_thread = threads.copy()

    # byr_hist as a decimal
    x_thread.loc[:, BYR_HIST] = x_thread.byr_hist.astype('float32') / 10

    # thread count, including current thread
    x_thread[THREAD_COUNT] = x_thread.index.get_level_values(level='thread')

    # reindex to create x_thread
    x_thread = pd.DataFrame(index=idx).join(x_thread)

    if turn_indicators:
        x_thread = add_turn_indicators(x_thread)

    return x_thread.astype('float32')


def get_arrival_times(clock, lstg_start, lstg_end, append_last=False):
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


def calculate_remaining(lstg_start=None, clock=None, idx=None):
    # start of delay period
    delay_start = clock.groupby(
        ['lstg', 'thread']).shift().dropna().astype('int64')
    # remaining is turn-specific
    remaining = pd.Series(1.0, index=idx)
    turns = idx.unique(level='index')
    for turn in turns:
        if turn > 1:
            turn_start = delay_start.xs(turn, level='index').reindex(index=idx)
            mask = idx.get_level_values(level='index') == turn
            remaining.loc[mask] = get_remaining(
                lstg_start, turn_start, MAX_DELAY[turn])
    # error checking
    assert np.all(remaining > 0) and np.all(remaining <= 1)
    return remaining


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


def get_x_offer_init(offers, idx, role=None, delay=None):
    # initialize dictionary of offer features
    x_offer = {}

    # dataframe of offer features for relevant threads
    threads = idx.droplevel(level='index').unique()
    offers = pd.DataFrame(index=threads).join(offers)

    # drop time feats from buyer models
    if role == BYR_PREFIX:
        offers.drop(TIME_FEATS, axis=1, inplace=True)

    # last index
    last = max(IDX[role])
    if delay:
        last -= 1

    # turn features
    for i in range(1, last + 1):
        # offer features at turn i, and turn number
        offer = offers.xs(i, level='index').reindex(
            index=idx, fill_value=0).astype('float32')
        turn = offer.index.get_level_values(level='index')

        # all features are zero for future turns
        offer.loc[i > turn, :] = 0.

        # current turn features
        if not delay and role == SLR_PREFIX:
            assert (offer.loc[i == turn, AUTO] == 0).all()
            assert (offer.loc[i == turn, EXP] == 0).all()
            offer.loc[i == turn, [CON, NORM, SPLIT, REJECT, MSG]] = 0.
        else:
            offer.loc[i == turn, :] = 0.

        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')

    # error checking
    check_zero(x_offer)

    return x_offer


def construct_x_init(part=None, role=None, delay=None, idx=None,
                     offers=None, threads=None, clock=None,
                     lstg_start=None):
    # delay must be True for byr
    if role == BYR_PREFIX:
        assert delay

    # listing features
    x = init_x(part, idx)

    # thread features
    x_thread = get_x_thread(threads, idx, turn_indicators=True)

    if role == BYR_PREFIX:
        # split master index
        idx1 = pd.Series(index=idx).xs(
            1, level='index', drop_level=False).index
        idx2 = idx.drop(idx1)

        # remove auto accept/reject features from x['lstg'] for buyer models
        x['lstg'].drop(['auto_decline', 'auto_accept',
                        'has_decline', 'has_accept'],
                       axis=1, inplace=True)

        # current time feats
        clock1 = pd.Series(DAY * idx1.get_level_values(level='day'),
                           index=idx1) + lstg_start.reindex(index=idx1,
                                                            level='lstg')
        clock2 = clock.xs(1, level='index').reindex(index=idx2)
        date_feats = pd.concat([extract_day_feats(clock1),
                                extract_day_feats(clock2)]).sort_index()
        date_feats.rename(lambda c: 'thread_{}'.format(c),
                          axis=1, inplace=True)

        # thread features
        x_thread = date_feats.join(x_thread)

        # redefine months_since_lstg
        x_thread.loc[idx1, MONTHS_SINCE_LSTG] = \
            idx1.get_level_values(level='day') * DAY / MONTH

    # remaining
    if delay:
        x_thread[INT_REMAINING] = \
            calculate_remaining(lstg_start=lstg_start,
                                clock=clock,
                                idx=idx)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer_init(offers, idx, role=role, delay=delay))

    return x


def get_sale_norm(offers):
    is_sale = offers[CON] == 1.
    norm = offers.loc[is_sale, NORM]
    # for seller, norm is defined as distance from start_price
    slr_turn = norm.index.isin(IDX[SLR_PREFIX], level='index')
    norm_slr = 1 - norm.loc[slr_turn]
    norm.loc[slr_turn] = norm_slr
    return norm


def get_policy_data(part):
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')
    clock = load_file(part, 'clock')
    lstg_start = load_file(part, 'lookup')[START_TIME]
    return offers, threads, clock, lstg_start


def get_value_data(part):
    # lookup file
    lookup = load_file(part, 'lookup')
    # load simulated data
    threads = load_file(part, 'x_thread_sim')
    offers = load_file(part, 'x_offer_sim')
    clock = load_file(part, 'clock_sim')
    # drop listings with infrequent arrivals
    lookup = lookup.loc[lookup[NO_ARRIVAL] < NO_ARRIVAL_CUTOFF, :]
    threads = threads.reindex(index=lookup.index, level='lstg')
    offers = offers.reindex(index=lookup.index, level='lstg')
    clock = clock.reindex(index=lookup.index, level='lstg')
    return lookup, threads, offers, clock


def save_featnames(x, m):
    """
    Creates dictionary of input feature names.
    :param x: dictionary of input dataframes.
    :param m: string name of model.
    """
    # initialize featnames dictionary
    featnames = {k: list(v.columns) for k, v in x.items() if 'offer' not in k}

    # for offer models
    if 'offer1' in x:
        # buyer models do not have time feats
        if BYR_PREFIX in m or m[-1] in [str(i) for i in IDX[BYR_PREFIX]]:
            feats = CLOCK_FEATS + OUTCOME_FEATS
        else:
            feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS

        # check that all offer groupings have same organization
        for k in x.keys():
            if 'offer' in k:
                assert list(x[k].columns) == feats

        # one vector of featnames for offer groupings
        featnames['offer'] = feats

    dump(featnames, INPUT_DIR + 'featnames/{}.pkl'.format(m))


def save_sizes(x, m):
    """
    Creates dictionary of input sizes.
    :param x: dictionary of input dataframes.
    :param m: string name of model.
    """
    sizes = dict()

    # count components of x
    sizes['x'] = {k: len(v.columns) for k, v in x.items()}

    # for arrival models, save interval and interval counts
    if ARRIVAL_PREFIX in m:
        sizes['interval'] = INTERVAL[1]
        sizes['interval_count'] = INTERVAL_COUNTS[1]
    elif m in DELAY_MODELS:
        turn = int(m[-1])
        sizes['interval'] = INTERVAL[turn]
        sizes['interval_count'] = INTERVAL_COUNTS[turn]

    # for init models, save discount rate
    if m in INIT_VALUE_MODELS:
        sizes[MONTHLY_DISCOUNT] = DELTA_MONTH
        sizes[ACTION_DISCOUNT] = DELTA_ACTION
        sizes[ACTION_COST] = C_ACTION

    # length of model output vector
    sizes['out'] = NUM_OUT[m]

    dump(sizes, INPUT_DIR + 'sizes/{}.pkl'.format(m))


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
    convert_x_to_numpy(d['x'], idx)

    # convert outcome to numpy
    d['y'] = d['y'].to_numpy()

    # save data
    dump(d, INPUT_DIR + '{}/{}.gz'.format(part, name))

    # save index
    dump(idx, INDEX_DIR + '{}/{}.gz'.format(part, name))

    # save subset
    if part in [TRAIN_MODELS, TRAIN_RL]:
        save_small(d, name)
