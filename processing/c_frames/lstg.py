from collections import OrderedDict
import numpy as np
from processing.util import extract_day_feats, get_lstgs, do_rounding
from constants import PARTS_DIR, INPUT_DIR, DAY
from featnames import START_PRICE, META, LEAF, CNDTN, SLR, BYR, LSTG, X_LSTG, \
    VALIDATION, DEC_PRICE, ACC_PRICE
from utils import input_partition, topickle, load_feats

AS_IS_FEATS = ['store', 'slr_us', 'fast', 'photos', 'slr_lstg_ct',
               'slr_bo_ct', 'start_price_pctile', 'fdbk_score', 'fdbk_pstv']


def construct_lstg_feats(listings):
    # initialize output dataframe with as-is features
    df = listings[AS_IS_FEATS].copy()
    # binary feats
    df['has_photos'] = listings.photos > 0
    df['fdbk_100'] = listings.fdbk_pstv == 1
    df['start_is_round'], df['start_is_nines'] = \
        do_rounding(listings[START_PRICE])
    # normalize start_date to years
    df['start_years'] = listings.start_date / 365
    # date features
    date_feats = extract_day_feats(listings.start_date * DAY)
    df = df.join(date_feats.rename(lambda x: 'start_' + x, axis=1))
    # condition
    s = listings[CNDTN]
    df['new'] = s == 1
    df['used'] = s == 7
    df['refurb'] = s.isin([2, 3, 4, 5, 6])
    df['wear'] = s.isin([8, 9, 10, 11]) * (s - 7)
    # auto decline/accept prices
    df['auto_decline'] = listings[DEC_PRICE] / listings[START_PRICE]
    df['has_decline'] = listings[DEC_PRICE] > 0
    df['auto_accept'] = listings[ACC_PRICE] / listings[START_PRICE]
    df['has_accept'] = listings[ACC_PRICE] < listings[START_PRICE]
    # remove slr prefix
    df.rename(lambda c: c[4:] if c.startswith('slr_') else c, 
              axis=1, inplace=True)
    return df.astype('float32')


def create_x_lstg(lstgs=None):
    # listing features
    listings = load_feats('listings', lstgs=lstgs)

    # initialize output dictionary
    x = dict()

    # listing features
    x[LSTG] = construct_lstg_feats(listings)

    # word2vec features
    for role in [BYR, SLR]:
        w2v = load_feats('w2v_{}'.format(role),
                         lstgs=listings[[LEAF]].values.squeeze(),
                         fill_zero=True)
        w2v.set_index(lstgs, inplace=True)
        x['w2v_{}'.format(role)] = w2v.astype('float32')
    del listings

    # slr and cat features
    for name in [SLR, META, LEAF]:
        x[name] = load_feats(name,
                             lstgs=lstgs,
                             fill_zero=True).astype('float32')

    # take natural log of number of listings
    for k, v in x.items():
        count_cols = [c for c in v.columns if c.endswith('lstgs')]
        for c in count_cols:
            x[k].loc[:, c] = x[k][c].apply(np.log1p)
            x[k].rename({c: c.replace('lstgs', 'ln_lstgs')},
                        axis=1, inplace=True)

    # ensure indices are aligned with lookup
    for v in x.values():
        assert np.all(lstgs == v.index)

    return x


def main():
    part = input_partition()
    print('{}/{}'.format(part, X_LSTG))

    # create dataframe
    x_lstg = create_x_lstg(lstgs=get_lstgs(part))

    # extract column names and save
    if part == VALIDATION:
        cols = OrderedDict()
        for k, v in x_lstg.items():
            cols[k] = list(v.columns)
        topickle(cols, INPUT_DIR + 'featnames/{}.pkl'.format(X_LSTG))

    # convert to numpy and save
    x_lstg = {k: v.values for k, v in x_lstg.items()}
    topickle(x_lstg, PARTS_DIR + '{}/{}.pkl'.format(part, X_LSTG))


if __name__ == "__main__":
    main()
