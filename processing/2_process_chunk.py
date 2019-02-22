"""
For each chunk of data, create simulator inputs.
"""

import argparse, pickle
import numpy as np, pandas as pd

COLNAMES = ['b1','s1','b2','s2','b3','s3','b4']

def getConcessions(O,T,thread_ids):
    df = pd.DataFrame(index=thread_ids, columns=COLNAMES)
    for i in thread_ids:
        prefix = np.array([0, T.start_price.loc[i]])
        prices = np.append(prefix, O.price.loc[i].values)
        for j in range(2,len(prices)):
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
    chunk = pickle.load(open(path, 'rb'))
    L, T, O = [chunk[k] for k in['listings','threads','offers']]
    T = T.join(L.loc[:,['start_price','bin_rev']], on='lstg')
    T = T.join(O.price.loc[pd.IndexSlice[:,1]])

    # ids of threads to keep
    thread_ids = T[(T.bin_rev == 0) & (T.price < T.start_price)].index

    # time-varying features

    # concessions
    concessions = getConcessions(O,T,thread_ids)
    slr_concessions = concessions.loc[:,['s1','s2','s3']]

    # offer features

    # constant features
    const_feats = T.start_price

    # seller turns
    slr_turns = slr_concessions.count()

    # write chunk
    chunk = {'offer_feats': offer_feats,
             'const_feats': const_feats,
             'slr_turns': slr_turns,
             'slr_concessions': slr_concessions}
    path = 'data/chunks/%d_simulator.pkl' % num
    pickle.dump(chunk, open(path, 'wb'))