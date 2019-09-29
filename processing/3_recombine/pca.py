import sys
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse, pickle
from datetime import datetime as dt
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from constants import *


def get_w2v(lstgs, role):
    # create lstg-category map
    lookup = lstgs[['meta', 'leaf', 'product']]
    # read in vectors
    w2v = pd.read_csv(W2V_PATH(role), index_col=0)
    # hierarchical join
    df = pd.DataFrame(np.nan, index=lookup.index, columns=w2v.columns)
    for level in ['product', 'leaf', 'meta']:
        mask = np.isnan(df[role[0] + '0'])
        df[mask] = lookup.loc[mask[mask].index, level].rename(
            'category').to_frame().join(w2v, on='category').drop(
            'category', axis=1)
    return df


def do_pca(df):
    df = StandardScaler().fit_transform(df)
    pca = PCA(n_components=len(df.columns), svd_solver='full')
    df = pca.fit_transform(df)
    return df


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load dataframes
    print('Loading data')
    lstgs = load_frames('lstgs')
    tf_slr = load_frames('tf_slr')
    tf_meta = load_frames('tf_meta')

    # time features
    print('Creating time features')
    tf_lstg = do_pca(tf_lstg)
    pickle.dump(tf_lstg, open(FRAMES_DIR + 'tf_lstg.pkl', 'wb'))

    # word2vec features
    w2v = get_w2v(lstgs, 'slr').join(get_w2v(lstgs, 'byr'))
    w2v = do_pca(w2v)
    pickle.dump(w2v, open(FRAMES_DIR + 'w2v_lstg.pkl', 'wb'))