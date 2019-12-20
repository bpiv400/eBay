import sys
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import input_partition
from processing.processing_utils import get_partition


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
    df['start_years'] = L.start_time / (365 * 24 * 3600)
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
    idx, path = get_partition(part)

    # listing features
    L = load(CLEAN_DIR + 'listings.pkl').reindex(index=idx)

    # listing features
    print('Listing features')
    lstg = get_x_lstg(L)
    dump(lstg, path('x_lstg'))
    del lstg

    # word2vec features
    print('Word2Vec features')
    for role in ['byr', 'slr']:
        w2v = load(W2V_DIR + '%s.gz' % role).reindex(
            index=L[['cat']].values.squeeze(), fill_value=0)
        w2v.set_index(L.index, inplace=True)
        dump(w2v, path('x_w2v_' + role))