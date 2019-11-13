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
    df = L[ASIS_FEATS]
    # normalize start_date to years
    df['start_years'] = L['start_date'] / 365
    # photos divided by 12, and binary indicator
    df['photos'] = L['photos'] / 12
    df['has_photos'] = L['photos'] > 0
    # slr feedback
    df['fdbk_100'] = df['fdbk_pstv'] == 1
    # prices
    df['decline'] = L['decline_price'] / L['start_price']
    df['accept'] = L['accept_price'] / L['start_price']
    for z in ['start', 'decline', 'accept']:
        df['is_round'], df['is_nines'] = do_rounding(L[z + '_price'])
    df['has_decline'] = df['decline'] > 0
    df['has_accept'] = df['accept'] < 1
    df['auto_dist'] = df['accept'] - df['decline']
    # condition
    s = L['cndtn']
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

    # initialize listing features
    x_lstg = get_x_lstg(L)

    # save
    dump(x_lstg, path('x_lstg'))
