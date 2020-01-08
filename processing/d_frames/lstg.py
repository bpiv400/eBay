import os, sys, h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, load_frames, \
    get_partition
from processing.processing_consts import CLEAN_DIR, W2V_DIR
from constants import *


# returns booleans for whether offer is round and ends in nines
def do_rounding(offer):
    digits = np.ceil(np.log10(offer.clip(lower=0.01)))
    factor = 5 * np.power(10, digits-3)
    diff = np.round(offer / factor) * factor - offer
    is_round = diff == 0
    is_nines = (diff > 0) & (diff <= factor / 5)
    return is_round, is_nines


def get_x_lstg(L):
    # initialize output dataframe with as-is features
    df = L[['store', 'slr_us', 'fast', 'slr_bos', 'slr_lstgs', \
            'fdbk_score', 'fdbk_pstv', 'start_price_pctile']].copy()
    # rename slr_bos and slr_lstgs
    df.rename({'slr_bos': 'slr_bos_total', 
        'slr_lstgs': 'slr_lstgs_total'}, axis=1, inplace=True)
    # normalize start_date to years
    df['start_years'] = L.start_date / 365
    # photos divided by 12, and binary indicator
    df['photos'] = L.photos / 12
    df['has_photos'] = L.photos > 0
    # slr feedback
    df['fdbk_100'] = df.fdbk_pstv == 1
    # prices
    df['auto_decline'] = L.decline_price / L.start_price
    df['auto_accept'] = L.accept_price / L.start_price
    for z in ['start', 'decline', 'accept']:
        df[z + '_is_round'], df[z + '_is_nines'] = do_rounding(L[z + '_price'])
    df['has_decline'] = df.auto_decline > 0
    df['has_accept'] = df.auto_accept < 1
    # condition
    s = L.cndtn
    df['new'] = s == 1
    df['used'] = s == 7
    df['refurb'] = s.isin([2, 3, 4, 5, 6])
    df['wear'] = s.isin([8, 9, 10, 11]) * (s - 7)
    return df


if __name__ == "__main__":
    # partition and corresponding indices
    part = input_partition()
    idx, _ = get_partition(part)

    # initialize output dictionary
    x = {}

    # listing features
    L = load(CLEAN_DIR + 'listings.pkl').reindex(index=idx)

    # listing features
    print('Listing features')
    x['lstg'] = get_x_lstg(L)

    # word2vec features
    print('Word2Vec features')
    for role in ['byr', 'slr']:
        w2v = load(W2V_DIR + '%s.gz' % role).reindex(
            index=L[['cat']].values.squeeze(), fill_value=0)
        w2v.set_index(L.index, inplace=True)
        x['w2v_{}'.format(role)] = w2v
    del L

    # slr features
    print('Seller features')
    x['slr'] = load_frames('slr').reindex(index=idx, fill_value=0)

    # cat and cndtn features
    print('Categorical features')
    df = load_frames('cat').reindex(index=idx, fill_value=0)
    for name in ['cat', 'cndtn']:
        x[name] = df[[c for c in df.columns if c.startswith(name + '_')]]

    # take natural log of number of listings
    for k, v in x.items():
        count_cols = [c for c in v.columns if c.endswith('_lstgs')]
        for c in count_cols:
            x[k].loc[:, c] = x[k][c].apply(np.log1p)
            x[k].rename({c: c.replace('lstgs', 'ln_lstgs')}, 
                axis=1, inplace=True)

    # ensure indices are aligned with lookup
    lookup = load(PARTS_DIR + '{}/lookup.gz'.format(part))
    for v in x.values():
        assert np.all(lookup.index == v.index)

    # featnames
    if part == 'train_models':
        featnames = {}
        for k, v in x.items():
            featnames[k] = list(v.columns)
        dump(featnames, INPUT_DIR + 'featnames/x_lstg.pkl')

    # numpy float32
    for k, v in x.items():
        x[k] = v.to_numpy(dtype='float32')

    # save hdf5 file
    with h5py.File(PARTS_DIR + '{}/x_lstg.hdf5'.format(part), 'w') as f:
        for k, v in x.items():
            f.create_dataset(k, data=v)