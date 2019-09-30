import sys
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from constants import *


def get_w2v(lstgs, role):
    # read in vectors
    w2v = pd.read_csv(W2V_PATH(role), index_col=0)
    # hierarchical join
    df = pd.DataFrame(np.nan, index=lstgs.index, columns=w2v.columns)
    for level in ['product', 'leaf', 'meta']:
        mask = np.isnan(df[role[0] + '0'])
        idx = mask[mask].index
        cat = lstgs[level].rename('category').reindex(index=idx).to_frame()
        df[mask] = cat.join(w2v, on='category').drop('category', axis=1)
    return df


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load dataframes
    print('Loading data')
    if num == 1:    # word2vec features
        name = 'w2v'
        # read in lstg ids
        ids = []
        for i in range(1, N_CHUNKS+1):
            ids += list(load(FEATS_DIR + str(i) + '_tf_slr.gz').index)
        ids = sorted(ids)
        # read in listings
        lstgs = pd.read_csv(CLEAN_DIR + 'listings.csv', index_col='lstg',
            usecols=['lstg', 'meta', 'leaf', 'product']).reindex(index=ids)
        # convert categories to strings
        for c in ['meta', 'leaf', 'product']:
            lstgs[c] = c[0] + lstgs[c].astype(str)
        mask = lstgs['product'] == 'p0'
        lstgs.loc[mask, 'product'] = lstgs.loc[mask, 'leaf']
        # replace with w2v vectors
        X = get_w2v(lstgs, 'slr').join(get_w2v(lstgs, 'byr'))
    elif num == 2:  # slr time features
        X = pd.DataFrame()
        name = 'tf_slr'
        for i in range(1, N_CHUNKS+1):
            stub = load(FEATS_DIR + str(i) + '_tf_slr.gz')
            X = X.append(stub)
    elif num == 3:  # meta time features
        X = pd.DataFrame()
        name = 'tf_meta'
        for i in range(N_META):
            stub = load(FEATS_DIR + 'm' + str(i) + '_tf_meta.gz')
            X = X.append(stub)
    else:
        print('%d:m incorrect id.' % num), exit()
    
    # standardize variables
    vals = StandardScaler().fit_transform(X)

    # PCA
    N = len(X.columns)
    pca = PCA(n_components=N, svd_solver='full')
    components = pca.fit_transform(vals)

    # select number of components
    shares = np.var(components, axis=0) / N
    keep = np.sum(shares >= PCA_CUTOFF)
    cols = [name + str(i) for i in range(1,keep+1)]
    out = pd.DataFrame(components[:,:keep], index=X.index, columns=cols)

    # save
    dump(out, PCA_DIR + name + '.gz')
