import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def get_x_lstg(lstgs):
    # initialize output dataframe with as-is features
    df = lstgs[BINARY_FEATS + COUNT_FEATS + ['start_date']]
    # clock features
    clock = pd.to_datetime(lstgs.start_date, unit='D', origin=START)
    df = df.join(extract_day_feats(clock))
    # slr feedback
    df.loc[df.fdbk_score.isna(), 'fdbk_score'] = 0
    df['fdbk_score'] = df.fdbk_score.astype(np.int64)
    df['fdbk_pstv'] = lstgs['fdbk_pstv'] / 100
    df.loc[df.fdbk_pstv.isna(), 'fdbk_pstv'] = 1
    df['fdbk_100'] = df['fdbk_pstv'] == 1
    # prices
    df['start'] = lstgs['start_price']
    df['decline'] = lstgs['decline_price'] / lstgs['start_price']
    df['accept'] = lstgs['accept_price'] / lstgs['start_price']
    for z in ['start', 'decline', 'accept']:
        df[z + '_round'], df[z +'_nines'] = do_rounding(df[z])
    df['has_decline'] = df['decline'] > 0
    df['has_accept'] = df['accept'] < 1
    df['auto_dist'] = df['accept'] - df['decline']
    # condition
    cndtn = lstgs['cndtn']
    df['new'] = cndtn == 1
    df['used'] = cndtn == 7
    df['refurb'] = cndtn.isin([2, 3, 4, 5, 6])
    df['wear'] = cndtn.isin([8, 9, 10, 11]) * (cndtn - 7)
    return df


def get_w2v(lstgs, role):
    # read in vectors
    w2v = pd.read_csv(W2V_PATH(role), index_col=0)
    # hierarchical join
    df = pd.DataFrame(np.nan, index=lstgs.index, columns=w2v.columns)
    for level in ['product', 'leaf', 'meta']:
        mask = np.isnan(df[role + '0'])
        idx = mask[mask].index
        cat = lstgs[level].rename('category').reindex(index=idx).to_frame()
        df[mask] = cat.join(w2v, on='category').drop('category', axis=1)
    return df


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # partition
    idx, path = get_partition(num)
    
    # load data 
    lstgs = load(CLEAN_DIR + 'listings.gz').drop(
        ['title', 'flag'], axis=1).reindex(index=idx)

    # lookup file
    print('lookup')
    lookup = lstgs[['meta', 'start_date', \
        'start_price', 'decline_price', 'accept_price']]
    dump(lookup, path('lookup'))

    # word2vec
    print('x_w2v')
    lstgs = categories_to_string(lstgs)
    w2v = get_w2v(lstgs, 'slr').join(get_w2v(lstgs, 'byr'))
    dump(w2v, path('x_w2v'))

    # listing features
    print('x_lstg')
    x_lstg = get_x_lstg(lstgs)
    dump(x_lstg, path('x_lstg'))
