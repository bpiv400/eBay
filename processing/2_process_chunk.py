"""
For each chunk of data, create simulator inputs.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from time_feats import get_time_feats

COLNAMES = ['b1', 's1', 'b2', 's2', 'b3', 's3', 'b4']


def add_turn_mask(df, targ=True):
    """
    Masks all rows where the response value is not defined
    by setting all values to -100

    Args:
        df: dataframe to be masked
    Kwargs:
        targ: boolean giving whether this is for seller replies
    Returns: df
    """
    if not targ:
        missing_rows = np.any(df.isna().values, axis=1)
        df.loc[missing_rows, :] = -100
    else:
        missing_rows = df.isna().values
        df[missing_rows] = -100
    return df


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


def get_offer_feats(concessions):
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
        [concessions.index.values, [1, 2, 3]], names=['thread_id', 'turn'])
    # create dataframe
    offer_feats = pd.DataFrame(index=index, columns=[
                               'con_slr', 'con_byr'])
    # prefix s0 column to concessions
    concessions['s0'] = 0
    for i in range(1, 3):
        prev = 's%d' % (i - 1)
        curr = 'b%d' % i
        pred = 's%d' % i
        # subset to thread ids where the response offer is defined
        threads = concessions.loc[~concessions[pred].isna(), pred].index
        index = pd.MultiIndex.from_product([threads, [i]])
        offer_feats.loc[index,
                        'con_slr'] = concessions.loc[threads, prev].values
        offer_feats.loc[index,
                        'con_byr'] = concessions.loc[threads, curr].values
    # add turn indicators
    offer_feats = add_turn_indicators(offer_feats)
    # add turn mask
    offer_feats = add_turn_mask(offer_feats, targ=False)
    # reshape data frame into a num_threads x num_turns x num_feats dataframe
    num_threads = len(offer_feats.index.levels[0])
    offer_feats = offer_feats.values.reshape(num_threads, 3, -1)
    return offer_feats


def getConcessions(O, T, thread_ids):
    df = pd.DataFrame(index=thread_ids, columns=COLNAMES)
    for i in thread_ids:
        prefix = np.array([0, T.start_price.loc[i]])
        prices = np.append(prefix, O.price.loc[i].values)
        for j in range(2, len(prices)):
            norm = (prices[j] - prices[j-2]) / (prices[j-1] - prices[j-2])
            df.loc[i, COLNAMES[j-2]] = norm
    return df


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    path = 'data/chunks/%d.pkl' % num
    print("Loading data")
    chunk = pickle.load(open(path, 'rb'))
    L, T, O = [chunk[k] for k in['listings', 'threads', 'offers']]
    T = T.join(L.loc[:, ['start_price', 'item', 'slr', 'bin_rev']], on='lstg')
    print("Done loading")
    # ids of threads to keep
    thread_ids = T[(T.bin_rev == 0) & (T.flag == 0)].index

    # time-varying features
    print("Getting time feats")
    time_feats = get_time_feats(O, T, L)

    # concessions
    print("Getting concessions")
    concessions = getConcessions(O, T, thread_ids)
    slr_concessions = concessions.loc[:, ['s1', 's2', 's3']]
    slr_concessions = add_turn_mask(slr_concessions)

    # offer features
    print("Getting concessions")
    offer_feats = get_offer_feats(concessions)

    # constant features
    print("Getting constant features")
    sim_const_feats = T.start_price.loc[thread_ids]
    sim_const_feats = np.expand_dims(sim_const_feats.values, 1)
    rl_const_feats = T.loc[thread_ids, ['start_price', 'item', 'slr']]

    # seller turns
    print("Getting seller turns")
    slr_turns = slr_concessions.count().values
    slr = T.slr.loc[thread_ids]

    # write simulator chunk
    print("Writing first chunk")
    chunk = {'offer_feats': offer_feats,
             'const_feats': sim_const_feats,
             'slr': slr,
             'slr_turns': slr_turns,  # NOTE not sure if this is necessary?
             'slr_concessions': slr_concessions}  # NOTE maybe we should change the name?
    path = 'data/chunks/%d_simulator.pkl' % num
    pickle.dump(chunk, open(path, 'wb'))

    # write rl chunk
    print("Writing second chunk")
    chunk = {'time_feats': time_feats,
             'rl_time': ['a'],
             'const_feats': rl_const_feats,
             'rl_const': ['start_price']
             }
    path = 'data/chunks/%d_rl.pkl' % num
    pickle.dump(chunk, open(path, 'wb'))
    print("Done")
