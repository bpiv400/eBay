"""
For each chunk of data, create simulator and RL inputs.
"""

import argparse, pickle
import numpy as np, pandas as pd
#from time_feats import get_time_feats


def get_slr_concessions(C):
    '''
    Creates a data frame of seller concessions with automatic accepts/rejects
    set to NA, and then removes threads for which all seller concessions are
    automatic.
    '''
    y = C[['s1', 's2', 's3']].copy()
    y.loc[C.s1_auto, 's1'] = np.nan
    y.loc[C.s2_auto, 's2'] = np.nan
    y.loc[C.s3_auto, 's3'] = np.nan
    # drop threads with no human slr concessions
    no_live_turns = pd.isna(y).sum(1) == 3
    y = y[~no_live_turns]
    return y


def add_turn_indicators(offer_feats):
    """
    Creates indicator for each turn
    """
    # turns
    turns = pd.DataFrame(0, index=[1, 2, 3], columns=['t1', 't2', 't3'])
    turns.index.name = 'turn'
    turns.loc[1, 't1'] = 1
    turns.loc[2, 't2'] = 1
    turns.loc[3, 't3'] = 1
    # join with offer features
    offer_feats = offer_feats.join(turns, on='turn')
    return offer_feats


def get_x_offer(C):
    """
    Creates a dataframe where each row gives the input to
    the simulator model at one step. Output variables:
        1. previous seller concession
        2. indicator for automatic rejection on last seller concession
        3. current buyer concession
        4. indicator for turn 1
        5. indicator for turn 2
        6. indicator for turn 3
    Index: [thread_id, turn] (turn in range 1...3)
    """
    # initialize data frame
    x_offer = pd.DataFrame(index=pd.MultiIndex.from_product(
        [C.index.values, [1, 2, 3]], names=['thread', 'turn']))
    # prefix s0 column to concessions
    C['s0'] = 0
    C['s0_auto'] = 0
    # loop over turns
    for i in range(1, 4):
        prev = 's%d' % (i - 1)
        prev_auto = 's%d_auto' % (i - 1)
        curr = 'b%d' % i
        pred = 's%d' % i
        # subset to thread ids where the response offer is defined
        threads = C.loc[~C[pred].isna(), pred].index
        index = pd.MultiIndex.from_product([threads, [i]])
        x_offer.loc[index, 'con_slr'] = C.loc[threads, prev].values
        x_offer.loc[index, 'auto'] = C.loc[threads, prev_auto].values
        x_offer.loc[index, 'con_byr'] = C.loc[threads, curr].values
    # add turn indicators
    x_offer = add_turn_indicators(x_offer)
    return x_offer


def get_x_fixed(T, thread_ids):
    # initialize dataframe
    fixed_feats = pd.DataFrame(index=thread_ids)
    # start_price
    fixed_feats = fixed_feats.join(T['start_price'])
    return fixed_feats


def get_concessions(O, T, thread_ids):
    '''
    Creates data frame of concessions for each buyer and seller turn.
    '''
    # create dataframe of clock times for each offer
    clock = O.loc[thread_ids]['clock'].unstack(level='index')
    # create dataframe of offers in dollars
    offers = O.loc[thread_ids]['price'].unstack(level='index')
    offers[0] = T.start_price.loc[thread_ids]
    # initialize dataframe for concessions
    C = pd.DataFrame(index=thread_ids)
    # first concession
    C['b1'] = offers[1] / offers[0]
    assert np.count_nonzero(np.isnan(C['b1'].values)) == 0
    # remaining concessions
    for i in range(2, 7):
        norm = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
        idx = 'b%d' % int((i+1) / 2) if i % 2 else 's%d' % int(i/2)
        C[idx] = norm
    # verify that all concessions are in bounds
    v = C.values.flatten()
    assert np.nanmax(v) <= 1 and np.nanmin(v) >= 0
    # auto accepts / rejects
    C['s1_auto'] = clock[2] == clock[1]
    C['s2_auto'] = clock[4] == clock[3]
    C['s3_auto'] = clock[6] == clock[5]
    return C


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    path = 'data/chunks/%d.pkl' % num
    print("Loading data")
    chunk = pickle.load(open(path, 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]
    T = T.join(L[['start_price', 'item', 'slr', 'bin_rev']], on='lstg')
    del chunk

    # ids of threads to keep
    thread_ids = T[(T.bin_rev == 0) & (T.flag == 0)].index.sort_values()

    # time-varying features
    print("Getting time feats")
    #time_feats = get_time_feats(O, T, L)

    # concessions
    print("Getting concessions")
    C = get_concessions(O, T, thread_ids)
    y = get_slr_concessions(C)

    # trim thread_ids
    thread_ids = y.index
    C = C.loc[thread_ids]

    # offer features
    print("Getting offer features")
    x_offer = get_x_offer(C)

    # constant features
    print("Getting constant features")
    x_fixed = get_x_fixed(T, thread_ids)
    rl_const_feats = T.loc[thread_ids, ['start_price', 'item', 'slr']]

    # write simulator chunk
    print("Writing first chunk")
    chunk = {'x_offer': x_offer,
             'x_fixed': x_fixed,
             'y': y,
             'slr': T.slr.loc[thread_ids]}
    path = 'data/chunks/%d_simulator.pkl' % num
    pickle.dump(chunk, open(path, 'wb'))

    # write rl chunk
    '''
    print("Writing second chunk")
    chunk = {'time_feats': time_feats,
             'rl_time': ['a'],
             'sim_time': ['a'],
             'const_feats': rl_const_feats,
             'rl_const': ['start_price'],
             'sim_const': ['start_price']
             }
    path = 'data/chunks/%d_rl.pkl' % num
    pickle.dump(chunk, open(path, 'wb'))
    print("Done")
    '''
