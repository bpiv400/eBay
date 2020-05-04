import numpy as np
import pandas as pd
from compress_pickle import dump
from inputs.inputs_consts import NUM_OUT, N_SMALL, INTERVAL, \
    INTERVAL_COUNTS, MONTHLY_DISCOUNT
from constants import INPUT_DIR, INDEX_DIR, VALIDATION, TRAIN_MODELS, \
    TRAIN_RL, IDX, BYR_PREFIX, TURN_FEATS, DELAY_MODELS, \
    INIT_VALUE_MODELS, INIT_MODELS, ARRIVAL_PREFIX, CON_MULTIPLIER
from featnames import CLOCK_FEATS, OUTCOME_FEATS, BYR_HIST, CON, \
    SPLIT, MSG, AUTO, EXP, REJECT, DAYS, DELAY, TIME_FEATS, THREAD_COUNT


def get_y_con(df, turn=None):
    # drop zero delay and expired offers
    mask = ~df[AUTO] & ~df[EXP]
    # concession is an int from 0 to 100
    y = (df.loc[mask, CON] * CON_MULTIPLIER).astype('int8')
    # boolean for accept in turn 7
    if turn == 7:
        y = y == 100
    return y


def get_x_thread(threads, idx):
    # initialize x_thread as copy
    x_thread = threads.copy()

    # byr_hist as a decimal
    x_thread.loc[:, BYR_HIST] = x_thread.byr_hist.astype('float32') / 10

    # thread count, including current thread
    x_thread[THREAD_COUNT] = x_thread.index.get_level_values(level='thread')

    # reindex to create x_thread
    x_thread = pd.DataFrame(index=idx).join(x_thread)

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

        # add turn indicators for RL initializations
        if m in INIT_MODELS:
            role = m.split('_')[-1]
            feats += TURN_FEATS[role]

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
        sizes['delta'] = MONTHLY_DISCOUNT

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
