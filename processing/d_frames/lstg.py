import sys, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from constants import *
from utils import *
from processing.processing_utils import *


# scales variables and performs PCA
def do_pca(df, pre):
    # standardize variables
    vals = StandardScaler().fit_transform(df)
    # PCA
    N = len(df.columns)
    pca = PCA(n_components=N, svd_solver='full')
    components = pca.fit_transform(vals)
    # return dataframe
    return pd.DataFrame(components, index=df.index, 
        columns=['%s%d' % (pre, i) for i in range(N)])


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
    df = L[ASIS_FEATS + ['start_date']]
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
    # listing features
    L = load(CLEAN_DIR + 'listings.pkl')
    L = L.loc[(L.flag == 0) & (L.toDrop == 0)]
    L = L.drop(['slr', 'meta', 'end_time', 'flag', 'toDrop', \
        'arrival_rate'], axis=1)
    cat = L[['cat']]
    idx = L.index

    # initialize listing features
    x_lstg = get_x_lstg(L)
    del L

    # pca on other dataframes
    for pre in ['w2v', 'slr', 'cat']:
        print(pre)
        # embeddings
        if pre == 'w2v':        
            for role in ['byr', 'slr']:
                w2v = load(W2V_DIR + '%s.gz' % role)
                cat = cat.join(w2v, on='cat')
            var = cat.drop('cat', axis=1)
        # slr and cat feats
        else:
            var = load_frames(pre).reindex(
                index=idx, fill_value=0)
        # pca and append
        var = do_pca(var, pre)
        x_lstg = x_lstg.join(var)
        del var

    # save by partition
    partitions = load(PARTS_DIR + 'partitions.gz')
    for part, indices in partitions.items():
        dump(x_lstg.reindex(index=indices), 
            PARTS_DIR + '%s/x_lstg.gz' % part)
