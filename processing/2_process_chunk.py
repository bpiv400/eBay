"""
For each chunk of data, create simulator and RL inputs.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from time_feats import get_time_feats

DIR = '../../data/chunks/'
END = 136079999


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print("Loading data")
    chunk = pickle.load(open(DIR + '%d.pkl' % num, 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]

    # time-varying features
    time_feats = get_time_feats(L, T, O)

    # ids of threads to keep
    T = T.join(L.drop(['bin', 'title', 'end_time'], axis=1))
    keep = (T.bin_rev == 0) & (T.flag == 0) & (L.end_time < END)
    T = T.loc[keep].drop(['start_time', 'bin_rev', 'flag'], axis=1)
    O = O.loc[keep][['price', 'message']]

    # write simulator chunk
    print("Writing chunk")
    chunk = {'O': O,
             'T': T,
             'time_feats': time_feats}
    pickle.dump(chunk, open(DIR + '%d_out.pkl' % num, 'wb'))
