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


def get_x_lstg(L):
    # initialize output dataframe with as-is features
    df = L[['store', 'slr_us', 'fast', 'slr_bos', 'slr_lstgs', \
            'fdbk_score', 'fdbk_pstv', 'start_price_pctile']]
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
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num-1

    # partition
    part = PARTITIONS[num]
    idx, path = get_partition(part)

    # listing features
    L = load(CLEAN_DIR + 'listings.pkl').reindex(index=idx)

    # listing features
    print('Listing features')
    df = get_x_lstg(L)

    # word2vec features
    print('Word2Vec features')
    for role in ['byr', 'slr']:
        w2v = load(W2V_DIR + '%s.gz' % role).reindex(
            index=L[['cat']].values.squeeze(), fill_value=0)
        w2v.set_index(L.index, inplace=True)
        df = df.join(w2v)
    del L, w2v

    # slr features
    print('Seller features')
    slr = load_frames('slr').reindex(index=idx, fill_value=0)
    df = df.join(slr)
    del slr

    # cat features
    print('Categorical features')
    cat = load_frames('cat').reindex(index=idx, fill_value=0)
    df = df.join(cat)

    # dump
    dump(df, path('x_lstg'))
    