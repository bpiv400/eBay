"""
Description of inputs:
    offers:
        cols: 'clock', 'price', 'message'
        index: thread, index
    listings:
        cols: 'meta', 'leaf', 'product', 'title', 'cndtn', 'slr', 'start_date',
            'end_date', 'relisted', 'fdbk_score', 'fdbk_pstv', 'start_price',
            'photos', 'slr_lstgs', 'slr_bos', 'views', 'wtchrs', 'sale_price',
            'ship_slow', 'ship_fast', 'ship_chosen', 'decline_price',
            'accept_price', 'bin_rev', 'store', 'slr_us', 'byr_us'
        index: 'lstg'
    threads:
        cols: 'lstg', 'byr', 'byr_us', 'byr_hist', 'slr_hist', 'start_time', 'flag'
        index: 'thread'
"""
import pickle
import pandas as pd
import numpy as np


def main():
    """
    Main method for subsampling to very small size for working locally
    """
    path = 'data/chunks/1.pkl'
    chunk = pickle.load(open(path, 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]
    slrs = L['slr'].unique()
    samps = int(len(slrs) / 20)
    slrs = np.random.choice(slrs, size=samps, replace=False)
    L = L.loc[L['slr'].isin(slrs), :]
    lstgs = L.index.values
    T = T.loc[T['lstg'].isin(lstgs), :]
    threads = T.index.values
    old_threads = O.index.get_level_values(0)
    subset = old_threads.isin(threads)
    O = O.loc[subset, :]
    chunk = {
        'listings': L,
        'threads': T,
        'offers': O
    }
    opath = 'data/chunks/2.pkl'
    pickle.dump(chunk, file=open(opath, 'wb'))


if __name__ == '__main__':
    main()
