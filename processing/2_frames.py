import sys, os
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse, pickle
import numpy as np, pandas as pd
from constants import *
from time_feats import *


def add_feats(events, L):
    # bool for byr turn
    events['byr'] = events.index.isin(IDX['byr'], level='index')
    # total concession
    events['norm'] = events.price / L.start_price
    events.loc[~events.byr, 'norm'] = 1 - events.norm
    # concession
    offers = events.price.drop(0, level='thread').unstack().join(
        L.start_price)
    offers = offers.rename({'start_price': 0}, axis=1)
    con = pd.DataFrame(index=offers.index)
    con[0] = 0
    con[1] = offers[1] / offers[0]
    for i in range(2, 8):
        con[i] = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
    events['con'] = con.stack()
    return events


def create_obs(df, isStart):
    toAppend = pd.DataFrame(index=df.index, columns=['clock', 'index',
        'censored', 'price', 'accept', 'reject', 'message', 'bin'])
    for c in ['accept', 'message', 'bin']:
        toAppend[c] = False
    if isStart:
        toAppend.loc[:, 'reject'] = False
        toAppend.loc[:, 'index'] = 0
        toAppend.loc[:, 'censored'] = False
        toAppend.loc[:, 'price'] = df.start_price
        toAppend.loc[:, 'clock'] = df.start_time
    else:
        toAppend.loc[:, 'reject'] = True
        toAppend.loc[:, 'index'] = 1
        toAppend.loc[:, 'censored'] = True
        toAppend.loc[:, 'price'] = np.nan
        toAppend.loc[:, 'clock'] = df.end_time
    return toAppend.set_index('index', append=True)


def add_start_end(offers, L):
    # listings dataframe
    lstgs = L[LEVELS[:6] + ['start_date', 'end_time', 'start_price']].copy()
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
    # sort
    return offers.sort_index()


def expand_index(df):
    df.set_index(LEVELS[:6], append=True, inplace=True)
    idxcols = LEVELS + ['thread']
    if 'index' in df.index.names:
        idxcols += ['index']
    df = df.reorder_levels(idxcols)
    df.sort_values(idxcols, inplace=True)
    return df


def init_offers(L, T, O):
    offers = O.join(T['start_time'])
    for c in ['accept', 'bin', 'reject', 'censored', 'message']:
        offers[c] = offers[c].astype(np.bool)
    offers['clock'] += offers['start_time']
    offers.drop('start_time', axis=1, inplace=True)
    offers = offers.join(L[LEVELS[:6]])
    offers = expand_index(offers)
    return offers


def create_events(L, T, O):
    # initial offers data frame
    offers = init_offers(L, T, O)
    # add start times and expirations for unsold listings
    events = add_start_end(offers, L)
    # add features for later use
    events = add_feats(events, L)
    # recode byr rejects that don't end thread
    idx = events.reset_index('index')[['index']].set_index(
        'index', append=True, drop=False).squeeze()
    tofix = events.byr & events.reject & (idx < idx.groupby(
        LEVELS + ['thread']).transform('max'))
    events.loc[tofix, 'reject'] = False
    return events


def get_multi_lstgs(L):
    df = L[LEVELS[:-1] + ['start_date', 'end_time']].set_index(
        LEVELS[:-1], append=True).reorder_levels(LEVELS).sort_index()
    # start time
    df['start_date'] *= 24 * 3600
    df = df.rename(lambda x: x.split('_')[0], axis=1)
    # find multi-listings
    df = df.sort_values(df.index.names[:-1] + ['start'])
    maxend = df.end.groupby(df.index.names[:-1]).cummax()
    maxend = maxend.groupby(df.index.names[:-1]).shift(1)
    overlap = df.start <= maxend
    return overlap.groupby(df.index.names).max()


def clean_events(events, L):
	# identify multi-listings
	ismulti = get_multi_lstgs(L)
	# drop multi-listings
	events = events[~ismulti.reindex(index=events.index)]
	# limit index to ['lstg', 'thread', 'index']
	events = events.reset_index(LEVELS[:-1], drop=True).sort_index()
	# 30-day burn in
	events = events.join(L['start_date'])
	events = events[events.start_date >= 30].drop('start_date', axis=1)
	# drop listings in which prices have changed
	events = events.join(L['flag'])
	events = events[events.flag == 0].drop('flag', axis=1)
	return events


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print('Loading data')
    outfile = CHUNKS_DIR + '%d_frames.pkl' % num
    chunk = pickle.load(open(CHUNKS_DIR + '%d.pkl' % num, 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]

    # categories to strings
    for c in ['meta', 'leaf', 'product']:
        L[c] = c[0] + L[c].astype(str)
    mask = L['product'] == 'p0'
    L.loc[mask, 'product'] = L.loc[mask, 'leaf']

    # create events dataframe
    print('Creating offer events.')
    events = create_events(L, T, O)

    # get upper-level time-valued features
    print('Creating hierarchical time features') 
    tf_hier = get_hierarchical_time_feats(events)

    # drop flagged lstgs
    print('Restricting observations')
    events = clean_events(events, L)

    # split off listing events
    idx = events.xs(0, level='index').reset_index(
        'thread', drop=True).index
    lstgs = pd.DataFrame(index=idx).join(L.drop('flag', axis=1)).join(
        tf_hier.rename(lambda x: 'tf_' + x, axis=1))   
    events = events.drop(0, level='thread') # remove lstg start/end obs

    # split off threads dataframe
    events = events.join(T[['byr_hist', 'byr_us']])
    threads = events[['clock', 'byr_us', 'byr_hist', 'bin']].xs(
        1, level='index')
    events = events.drop(['byr_us', 'byr_hist', 'bin'], axis=1)

    # exclude current thread from byr_hist
    threads['byr_hist'] -= (1-threads.bin) 

    # write chunk
    print("Writing chunk")
    chunk = {'events': events, 'lstgs': lstgs, 'threads': threads}
    pickle.dump(chunk, open(outfile, 'wb'))