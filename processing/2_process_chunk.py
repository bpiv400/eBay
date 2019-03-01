"""
For each chunk of data, create simulator and RL inputs.
"""

import argparse, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn
#from time_feats import get_time_feats


def convert_to_tensors(x_offer, x_fixed, y):
    tensors = {}

    # order threads by sequence length
    threads = (y.isnull().sum(1)).sort_values(ascending=True).index

    # seller concessions
    M_y = np.transpose(y.loc[threads].values)
    tensors['y'] = torch.tensor(np.reshape(M_y, (3, len(threads), 1))).float()

    # fixed features
    M_fixed = np.reshape(x_fixed.loc[threads].values, (1, len(threads), -1))
    tensors['x_fixed'] = torch.tensor(M_fixed).float()

    # offer features
    dfs = [x_offer.xs(t, level='turn').loc[threads] for t in range(1, 4)]
    arrays = [dfs[t].values.astype(float) for t in range(3)]
    reshaped = [np.reshape(arrays[t], (1, len(threads), -1)) for t in range(3)]
    ttuple = tuple([torch.tensor(reshaped[i]) for i in range(3)])
    tensors['x_offer'] = torch.cat(ttuple, 0).float()

    return tensors, threads


def add_turn_indicators(offer_feats):
    """
    Creates an indicator for each turn
    """
    # initialize table of indicators
    turns = pd.DataFrame(0, index=[1, 2, 3], columns=['t1', 't2', 't3'])
    turns.index.name = 'turn'
    turns.loc[1, 't1'] = 1
    turns.loc[2, 't2'] = 1
    turns.loc[3, 't3'] = 1
    # join with offer features
    offer_feats = offer_feats.join(turns, on='turn')
    return offer_feats


def get_offer_feats(C):
    """
    Creates a dataframe where each row gives the input to
    the simulator model at one step

    Columns: [previous seller concession, current buyer concession, [set of 3 turn indicators]]
    Index: [thread_id, turn] (turn in range 1...3)

    For each timestep where at least 1 input or the response variable does not exist, we leave
    the corresponding row as a row of NAs
    """
    # create index
    index = pd.MultiIndex.from_product(
        [C.index.values, [1, 2, 3]], names=['thread', 'turn'])
    # create dataframe
    offer_feats = pd.DataFrame(index=index, columns=['con_slr', 'con_byr'])
    # prefix s0 column to concessions
    C['s0'] = 0
    for i in range(1, 4):
        prev = 's%d' % (i - 1)
        curr = 'b%d' % i
        pred = 's%d' % i
        # subset to thread ids where the response offer is defined
        threads = C.loc[~C[pred].isna(), pred].index
        index = pd.MultiIndex.from_product([threads, [i]])
        offer_feats.loc[index, 'con_slr'] = C.loc[threads, prev].values
        offer_feats.loc[index, 'con_byr'] = C.loc[threads, curr].values
    # add turn indicators
    offer_feats = add_turn_indicators(offer_feats)
    return offer_feats


def get_fixed_feats(T, thread_ids):
    # initialize dataframe
    fixed_feats = pd.DataFrame(index=thread_ids)
    # start_price
    fixed_feats = fixed_feats.join(T['start_price'])
    return fixed_feats


def getConcessions(O, T, thread_ids):
    '''
    Creates data frame of concessions for each buyer and seller turn.
    '''
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
    return C


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num - 1

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
    C = getConcessions(O, T, thread_ids)

    # offer features
    print("Getting offer features")
    x_offer = get_offer_feats(C)

    # constant features
    print("Getting constant features")
    x_fixed = get_fixed_feats(T, thread_ids)
    rl_const_feats = T.loc[thread_ids, ['start_price', 'item', 'slr']]

    # seller concessions
    y = C[['s1', 's2', 's3']]

    # write simulator chunk
    print("Writing first chunk")
    tensors, threads = convert_to_tensors(x_offer, x_fixed, y)
    chunk = {'x_offer': tensors['x_offer'],
             'x_fixed': tensors['x_fixed'],
             'y': tensors['y'],
             'slr': T.slr.loc[thread_ids],
             'threads': threads}
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
