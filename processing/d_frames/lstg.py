import sys, argparse
from compress_pickle import load, dump
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


# scales variables and performs PCA
def do_pca(df):
    # standardize variables
    vals = StandardScaler().fit_transform(df)
    # PCA
    N = len(df.columns)
    pca = PCA(n_components=N, svd_solver='full')
    components = pca.fit_transform(vals)
    # return dataframe
    return pd.DataFrame(components, index=df.index, 
        columns=['c' + str(i) for i in range(N)])


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
    df = lstgs[ASIS_FEATS + ['start_date']]
    # photos divided by 12, and binary indicator
    df.loc[:, 'photos'] = lstgs['photos'] / 12
    df.loc[:, 'has_photos'] = lstgs['photos'] > 0
    # slr feedback
    df.loc[:, 'fdbk_100'] = df['fdbk_pstv'] == 1
    # prices
    df.loc[:, 'start'] = lstgs['start_price_pctile']
    df.loc[:, 'decline'] = lstgs['decline_price'] / lstgs['start_price']
    df.loc[:, 'accept'] = lstgs['accept_price'] / lstgs['start_price']
    for z in ['start', 'decline', 'accept']:
        is_round, is_nines = do_rounding(lstgs[z + '_price'])
        df.loc[:, z + '_round'] = is_round
        df.loc[:, z +'_nines'] = is_nines
    df['has_decline'] = df['decline'] > 0
    df['has_accept'] = df['accept'] < 1
    df['auto_dist'] = df['accept'] - df['decline']
    # condition
    s = lstgs['cndtn']
    df['new'] = s == 1
    df['used'] = s == 7
    df['refurb'] = s.isin([2, 3, 4, 5, 6])
    df['wear'] = s.isin([8, 9, 10, 11]) * (s - 7)
    return df


def get_w2v(lstgs, role):
    # read in vectors
    w2v = load(W2V_PATH(role))
    # hierarchical join
    return lstgs[['cat']].join(w2v, on='cat').drop('cat', axis=1)


if __name__ == "__main__":
    # load partitions, concatenate indices
    partitions = load(PARTS_DIR + 'partitions.gz')
    idx = np.sort(np.concatenate(list(partitions.values())))

    # listing features
    lstgs = load(CLEAN_DIR + 'listings.gz').drop(
        ['title', 'flag'], axis=1).reindex(index=idx)
    x_lstg = get_x_lstg(lstgs)

    # add slr and byr embeddings
    cat = lstgs[['cat']]
    for role in ['byr', 'slr']:
        w2v = load(W2V_PATH(role))
        cat = cat.join(w2v, on='cat')
    x_lstg = x_lstg.join(cat.drop('cat', axis=1))
    del lstgs, w2v, cat

    # add slr features
    x_lstg = x_lstg.join(load_frames('slr').reindex(
        index=x_lstg.index, fill_value=0))

    # add cat features
    x_lstg = x_lstg.join(load_frames('cat').reindex(
        index=x_lstg.index, fill_value=0))

    # perform pca
    x_lstg = do_pca(x_lstg)

    # save by partition
    for part, idx in partitions.items():
        dump(x_lstg.reindex(index=idx), 
            PARTS_DIR + '%s/x_lstg.gz' % part)
