import sys, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


# returns booleans for whether offer is round and ends in nines
def do_rounding(offer):
    digits = np.ceil(np.log10(offer.clip(lower=0.01)))
    factor = 5 * np.power(10, digits-3)
    diff = np.round(offer / factor) * factor - offer
    is_round = diff == 0
    is_nines = (diff > 0) & (diff <= factor / 5)
    return is_round, is_nines


def get_x_lstg(lstgs):
    # initialize output dataframe with as-is features
    df = lstgs[BINARY_FEATS + COUNT_FEATS + ['start_date']]
    # slr feedback
    df['fdbk_100'] = df['fdbk_pstv'] == 1
    # prices
    df['start'] = lstgs['start_price_pctile']
    df['decline'] = lstgs['decline_price'] / lstgs['start_price']
    df['accept'] = lstgs['accept_price'] / lstgs['start_price']
    for z in ['start', 'decline', 'accept']:
        is_round, is_nines = do_rounding(df[z])
        df.loc[:, z + '_round'] = is_round
        df.loc[:, z +'_nines'] = is_nines
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
    w2v = load(W2V_PATH(role))
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
    num = parser.parse_args().num-1

    # partition
    part = PARTITIONS[num]
    idx, path = get_partition(part)

    # load data 
    lstgs = load(CLEAN_DIR + 'listings.gz').drop(
        ['title', 'flag'], axis=1).reindex(index=idx)
    tf_slr = load_frames('tf_slr').reindex(index=idx)
    tf_meta = load_frames('tf_meta').reindex(index=idx)

    # lookup file
    lookup = lstgs[['meta', 'start_date', \
        'start_price', 'decline_price', 'accept_price']]
    dump(lookup, path('lookup'))

    # listing features
    x_lstg = get_x_lstg(lstgs)

    # add slr and byr embeddings
    lstgs = categories_to_string(lstgs)
    for role in ['byr', 'slr']:
        w2v = get_w2v(lstgs, role)
        x_lstg = x_lstg.join(w2v)

    # add slr features
    # x_lstg = x_lstg.join(tf_slr)

    # # add categorical features
    # tf_meta = []
    # for i in range(N_META):
    #     tf_meta.append(load(FEATS_DIR + 'm' + str(i) + '_tf_meta.gz'))
    # tf_meta = pd.concat(tf_meta).reindex(index=idx)
    # x_lstg = x_lstg.join(tf_meta)

    # save
    dump(x_lstg, path('x_lstg'))
