"""
For each chunk of data, create simulator and RL inputs.
"""

import argparse, pickle
from datetime import datetime as dt
import numpy as np, pandas as pd
from time_funcs import *

DIR = '../../data/chunks/'
LDADIR = '../../data/lda/'
END = 136079999
START_DATE = '2012-06-01'
START_TIME = '2012-06-01 00:00:00'
TOL_HALF = 0.02 # count concessions within this range as 1/2
LEVELS = ['slr', 'meta', 'leaf', 'title', 'cndtn', 'lstg']
COLS = ['delay', 'con', 'round', 'nines', 'msg']
META_OTHER = [0, 19]
LSTG_FEATS = ['relisted', 'store', 'slr_us', 'store', 'fast',
               'photos', 'slr_bos', 'slr_lstgs', 'fdbk_score', 'fdbk_pstv']
THREAD_FEATS = ['slr_hist', 'byr_hist', 'byr_us']
MAX_DELAY = {'slr': 2 * 24 * 3600, 'byr': 14 * 24 * 3600}
FUNC = {'slr': [open_lstgs, open_threads],
        'meta': [open_lstgs, open_threads],
        'leaf': [open_lstgs, open_threads],
        'title': [slr_min, byr_max,
                  slr_offers, byr_offers,
                  open_lstgs, open_threads],
        'cndtn': [slr_min, byr_max,
                  slr_offers, byr_offers,
                  open_lstgs, open_threads],
        'lstg': [slr_min, byr_max,
                 slr_offers, byr_offers,
                 open_threads]}
TFNAMES = ['_'.join([k, f.__name__]) for k, v in FUNC.items() for f in v]


def get_y_other(df, byr_hist):
    # output vectors
    delay = df[['delay', 'censored']].copy()
    con = df['con'].copy()
    digits = df['round'] + 2 * df['nines']
    msg = df['message'].copy()
    # variables to restrict by
    arrival = df.index.get_level_values(level='index') == 1
    auto = delay.delay == 0
    censored = delay.censored
    isbin = byr_hist == 0
    iscon = ~con.isin([0, 1])
    # apply restrictions
    delay = delay[~arrival & ~auto]
    con = con[~auto & ~censored & ~isbin]
    digits = digits[~auto & ~censored & ~isbin & iscon]
    msg = msg[~auto & ~censored & ~isbin & iscon]
    return delay, con, digits, msg


def get_y_arrivals(events):
    # restrict events to threads
    df = events.xs(1, level='index')
    # time difference between thread start and listing start
    start_date = pd.to_datetime(df.start_date.xs(1, level='thread'),
        unit='D', origin=START_DATE)
    ts = pd.to_datetime(df.clock, unit='s', origin=START_TIME)
    diff = ts - start_date
    # create dictionary
    d = {}
    days = diff.dt.days
    sec = diff.dt.seconds[df.bin == 0] / (24 * 3600 - 1)
    loc = df.byr_us[df.bin == 0]
    hist = df.byr_hist[df.bin == 0]
    return days, sec, loc, hist


def do_rounding(offer):
    digits = np.ceil(np.log10(offer.clip(lower=0.01)))
    factor = 5 * np.power(10, digits-3)
    diff = np.round(offer / factor) * factor - offer
    is_round = diff == 0
    is_nines = (diff > 0) & (diff <= factor / 5)
    return is_round, is_nines


def get_x_thread(events):
    # initialize dataframe
    df = events.xs(1, level='index')
    x = pd.DataFrame(index=df.index)
    # thread features
    for z in THREAD_FEATS:
        x[z] = df[z]
    return x


def get_x_lstg(events, tf):
    '''
    Constructs a dataframe of fixed features that are used to initialize the
    hidden state and the LSTM cell.
    '''
    # dataframes to draw values from
    lstgs = events.groupby(['slr', 'lstg']).first()
    tf0 = tf.xs(0, level='thread').xs(0, level='index').reset_index('clock')
    lda_weights = pickle.load(open(LDADIR + 'weights.pkl', 'rb'))
    # initialize output dataframe
    x = pd.DataFrame(index=lstgs.index)
    # week and day of week of listing
    ts = pd.to_datetime(tf0.clock, unit='s', origin=START_TIME)
    x['week'] = ts.dt.week
    for i in range(7):
        x['dow' + str(i)] = ts.dt.dayofweek == i
    # initial time-valued features
    tf0 = tf0.drop('clock', axis=1)
    for col in tf0.columns:
        if not col.startswith('lstg_'):
            x[col] = tf0[col]
    # prices
    for z in ['start', 'decline', 'accept']:
        x[z] = lstgs[z + '_price']
        x[z + '_round'], x[z +'_nines'] = do_rounding(x[z])
        if z != 'start':
            x[z + '_norm'] = x[z] / x['start']
    x['has_decline'] = x['decline_norm'] > 0
    x['has_accept'] = x['accept_norm'] < 1
    x['auto_dist'] = x['accept_norm'] - x['decline_norm']
    # features without transformations
    for z in LSTG_FEATS:
        x[z] = lstgs[z]
    # indicator for perfect feedback score
    x['fdbk_100'] = x['fdbk_pstv'] == 100
    # leaf LDA scores
    w = lda_weights[:, lstgs.leaf]
    for i in range(len(lda_weights)):
        x['lda' + str(i)] = w[i, :]
    # one-hot vector for meta
    x['meta0'] = lstgs['meta'].isin(META_OTHER)
    for i in range(1, 35):
        if i not in META_OTHER:
            x['meta' + str(i)] = lstgs['meta'] == i
    # condition
    cndtn = lstgs['cndtn']
    x['no_cndtn'] = cndtn == 0
    x['new'] = cndtn == 1
    x['used'] = cndtn == 7
    x['refurb'] = cndtn.isin([2, 3, 4, 5, 6])
    x['wear'] = cndtn.isin([8, 9, 10, 11]) * (cndtn - 7)
    return x


def get_x_offer(events):
    '''
    Creates dataframe of offer and time variables.
    '''
    # initialize dataframe
    df = pd.DataFrame()
    # unstack offers
    offers = events['price'].unstack()
    offers[0] = events['start_price'].groupby(['slr', 'lstg']).first()
    # raw offer
    df['offer'] = offers.stack().sort_index()
    # normalized offer
    df['norm'] = df.offer / offers[0]
    # change from previous offer within role
    change = pd.DataFrame(index=offers.index)
    change[0] = 0
    change[1] = 0
    for i in range(2, 8):
        change[i] = abs(offers[i] - offers[i-2])
    df['change'] = change.stack().sort_index()
    # gap between successive offers
    gap = pd.DataFrame(index=offers.index)
    gap[0] = 0
    for i in range(1, 8):
        gap[i] = abs(offers[i] - offers[i-1])
    df['gap'] = gap.stack().sort_index()
    # offer digits
    df['round'], df['nines'] = do_rounding(df.offer)
    # concession
    con = pd.DataFrame(index=offers.index)
    con[0] = 0
    con[1] = offers[1] / offers[0]
    for i in range(2, 8):
        con[i] = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
    df['con'] = con.stack()
    df['reject'] = df['con'] == 0
    df['split'] = np.abs(df['con'] - 0.5) < TOL_HALF
    # boolean variables
    for z in ['message', 'censored']:
        s = events[z].unstack()
        s[0] = 0
        df[z] = s.stack().sort_index().astype(np.bool)
    # seconds since last offer
    clock = events['clock'].unstack()
    clock[0] = (events['start_date'] * 24 * 3600).groupby(
        ['slr', 'lstg']).first()
    delay = pd.DataFrame(index=clock.index)
    delay[0] = 0
    for i in range(1, 8):
        delay[i] = clock[i] - clock[i-1]
    df['delay'] = delay.rename_axis(
        'index', axis=1).stack().astype(np.int64)
    df['auto'] = df.delay == 0
    df['exp'] = df.delay == MAX_DELAY['slr']
    # timestamps
    clock = clock.stack().sort_index().astype(np.int64)
    ts = pd.to_datetime(clock, unit='s', origin=START_TIME)
    df['week'] = ts.dt.week
    for i in range(7):
        df['dow' + str(i)] = ts.dt.dayofweek == i
    df['min'] = ts.dt.hour * 60 + ts.dt.minute
    return df


def add_time_feats(events):
    df = events.copy()
    # loop over hierarchy
    for i in range(len(LEVELS)):
        levels = LEVELS[: i+1]
        ordered = df.copy().sort_values(levels + ['clock'])
        name = levels[-1]
        f = FUNC[name]
        for j in range(len(f)):
            newname = '_'.join([name, f[j].__name__])
            print('\t%s' % newname)
            feat = f[j](ordered.copy(), levels)
            if f[j].__name__ in ['slr_min', 'byr_max']:
                feat = feat.div(L['start_price'])
                feat = feat.reorder_levels(
                    LEVELS + ['thread', 'index', 'clock'])
            events[newname] = feat
    return events


def create_obs(df, isStart):
    toAppend = pd.DataFrame(index=df.index, columns=['clock', 'index',
        'censored', 'price', 'accept', 'reject', 'message', 'bin'])
    toAppend.loc[:, ['accept', 'reject', 'message', 'bin']] = 0
    if isStart:
        toAppend.loc[:, ['index', 'censored']] = 0
        toAppend.loc[:, 'price'] = df.start_price
        toAppend.loc[:, 'clock'] = df.start_time
    else:
        toAppend.loc[:, ['index', 'censored']] = 1
        toAppend.loc[:, 'price'] = np.nan
        toAppend.loc[:, 'clock'] = df.end_time
    return toAppend.set_index('index', append=True)


def add_start_end(offers, L):
    # listings dataframe
    lstgs = L[LEVELS[:5] + ['start_date', 'end_time', 'start_price']].copy()
    lstgs['thread'] = 0
    lstgs.set_index('thread', append=True, inplace=True)
    lstgs = expand_index(lstgs)
    lstgs['start_time'] = lstgs.start_date * 60 * 60 * 24
    lstgs.drop('start_date', axis=1, inplace=True)
    lstgs = lstgs.join(offers['accept'].groupby('lstg').max())
    # create data frames to append to offers
    start = create_obs(lstgs, True)
    end = create_obs(lstgs[lstgs.accept != 1], False)
    # append to offers
    offers = offers.append(start, sort=True).append(end, sort=True)
    # put clock into index
    offers = offers.set_index('clock', append=True)
    # sort
    return offers.sort_index()


def expand_index(df):
    df.set_index(LEVELS[:5], append=True, inplace=True)
    idxcols = LEVELS + ['thread']
    if 'index' in df.index.names:
        idxcols += ['index']
    df = df.reorder_levels(idxcols)
    df.sort_values(idxcols, inplace=True)
    return df


def init_offers(L, T, O):
    offers = O.join(T['start_time'])
    offers['clock'] += offers['start_time']
    offers.drop('start_time', axis=1, inplace=True)
    offers = offers.join(L[LEVELS[:5]])
    offers = expand_index(offers)
    return offers


def configure_events(L, T, O):
    # initial offers data frame
    offers = init_offers(L, T, O)
    # add start times and expirations for unsold listings
    events = add_start_end(offers, L)
    # add time valued features
    events = add_time_feats(events)
    # add listing features
    events = events.join(L.drop(LEVELS[:-1], axis=1))
    # limit index to ['slr', 'lstg', 'thread', 'clock', 'index']
    events = events.reset_index('title', drop=True).reset_index(
        ['meta', 'leaf', 'cndtn'])
    # drop lstg expiration events
    events = events[~events.price.isna()]
    # drop listings in which prices have changed
    events = events[events.flag == 0].drop('flag', axis=1)
    # drop right-censored listings
    events = events[events.end_time < END]
    # drop listings from 2012
    date = pd.to_datetime(events.start_date, unit='D', origin=START_DATE)
    events = events[date.dt.year == 2013].sort_index()
    # split into events and time features
    tf = events[TFNAMES]
    keep = [c for c in events.columns if c not in tf.columns]
    events = events[keep].drop(0, level='thread').reset_index('clock')
    # and thread features and clean up
    events = events.join(T[THREAD_FEATS]).reorder_levels(
        ['slr', 'lstg', 'thread', 'index']).sort_index()
    return events, tf


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print("Loading data")
    chunk = pickle.load(open(DIR + '%d.pkl' % num, 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]

    # events and time-varying features
    print('Creating events and time features')
    events, tf = configure_events(L, T, O)

    # input features
    print('Creating input features')
    x = {}
    x['thread'] = get_x_thread(events)
    x['lstg'] = get_x_lstg(events, tf)
    x['offer'] = get_x_offer(events)

    # outcome variables
    print('Creating outcome variables')
    y = {}
    y['days'], y['sec'], y['loc'], y['hist'] = get_y_arrivals(events)
    y['delay'], y['con'], y['digits'], y['message'] = get_y_other(
        x['offer'], events.byr_hist)

    # remove censored variable after creating outcomes
    x['offer'].drop('censored', axis=1, inplace=True)

    # write simulator chunk
    print("Writing chunk")
    chunk = {'L': L,
             'tf': tf,
             'y': y,
             'x': x}
    pickle.dump(chunk, open(DIR + '%d_out.pkl' % num, 'wb'))
