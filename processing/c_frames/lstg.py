import os
from compress_pickle import load, dump
import numpy as np
import pandas as pd
from processing.util import extract_day_feats
from utils import input_partition, load_file
from constants import CLEAN_DIR, W2V_DIR, PARTS_DIR, DAY, BYR_PREFIX, \
    SLR_PREFIX, NUM_CHUNKS
from featnames import START_PRICE

AS_IS_FEATS = ['store', 'slr_us', 'fast', 'photos', 'slr_lstg_ct',
               'slr_bo_ct', 'start_price_pctile', 'fdbk_score', 'fdbk_pstv']


def create_chunks(lookup, x, chunk_dir):
    lookup = lookup.sort_values(by=START_PRICE)
    # concatenate into one dataframe
    x = pd.concat(x.values(), axis=1)
    assert x.isna().sum().sum() == 0
    # iteration prep
    idx = np.arange(0, len(x), step=NUM_CHUNKS)
    # create chunks
    for i in range(NUM_CHUNKS):
        # create chunk and save
        chunk = {'lookup': lookup.iloc[idx, :],
                 'x_lstg': x.iloc[idx, :]}
        path = chunk_dir + '{}.gz'.format(i+1)
        dump(chunk, path)

        # increment indices
        idx = idx + 1
        if idx[-1] >= len(x):
            idx = idx[:-1]


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
    df = lstgs[AS_IS_FEATS].copy()
    # binary feats
    df['has_photos'] = lstgs.photos > 0
    df['fdbk_100'] = lstgs.fdbk_pstv == 1
    df['start_is_round'], df['start_is_nines'] = \
        do_rounding(lstgs.start_price)
    # normalize start_date to years
    df['start_years'] = lstgs.start_date / 365
    # date features
    date_feats = extract_day_feats(lstgs.start_date * DAY)
    df = df.join(date_feats.rename(lambda x: 'start_' + x, axis=1))
    # condition
    s = lstgs.cndtn
    df['new'] = s == 1
    df['used'] = s == 7
    df['refurb'] = s.isin([2, 3, 4, 5, 6])
    df['wear'] = s.isin([8, 9, 10, 11]) * (s - 7)
    # auto decline/accept prices
    df['auto_decline'] = lstgs.decline_price / lstgs.start_price
    df['auto_accept'] = lstgs.accept_price / lstgs.start_price
    df['has_decline'] = df.auto_decline > 0
    df['has_accept'] = df.auto_accept < 1 
    # remove slr prefix
    df.rename(lambda c: c[4:] if c.startswith('slr_') else c, 
              axis=1, inplace=True)
    return df


def main():
    # partition and corresponding indices
    part = input_partition()
    print('{}/x_lstg'.format(part))

    # lstg indices
    lookup = load_file(part, 'lookup')
    idx = lookup.index

    # listing features
    lstgs = load(CLEAN_DIR + 'listings.pkl').reindex(index=idx)

    # save end time
    dump(lstgs.end_time, PARTS_DIR + '{}/lstg_end.gz'.format(part))

    # initialize output dictionary
    x = dict()

    # listing features
    print('Listing features')
    x['lstg'] = get_x_lstg(lstgs)

    # word2vec features
    print('Word2Vec features')
    for role in [BYR_PREFIX, SLR_PREFIX]:
        w2v = load(W2V_DIR + '{}.gz'.format(role)).reindex(
            index=lstgs[['leaf']].values.squeeze(), fill_value=0)
        w2v.set_index(lstgs.index, inplace=True)
        x['w2v_{}'.format(role)] = w2v.astype('float32')
    del lstgs

    # slr and cat features
    print('Seller features')
    for name in ['slr', 'meta', 'leaf']:
        x[name] = load(PARTS_DIR + '{}/{}.gz'.format(part, name)).reindex(
            index=idx, fill_value=0).astype('float32')

    # take natural log of number of listings
    for k, v in x.items():
        count_cols = [c for c in v.columns if c.endswith('lstgs')]
        for c in count_cols:
            x[k].loc[:, c] = x[k][c].apply(np.log1p)
            x[k].rename({c: c.replace('lstgs', 'ln_lstgs')}, 
                        axis=1, inplace=True)

    # ensure indices are aligned with lookup
    for v in x.values():
        assert np.all(idx == v.index)

    # save as gz
    dump(x, PARTS_DIR + '{}/x_lstg.gz'.format(part))

    # make chunk directory
    chunk_dir = PARTS_DIR + '{}/chunks/'.format(part)
    if not os.path.isdir(chunk_dir):
        os.mkdir(chunk_dir)

    # save chunks
    create_chunks(lookup, x, chunk_dir)


if __name__ == "__main__":
    main()
