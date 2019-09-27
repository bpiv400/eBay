import sys
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse, pickle
from datetime import datetime as dt
from sklearn.utils.extmath import cartesian
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from constants import *
from time_feats import *
from utils import *


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


def get_x_lstg(lstgs):
    '''
    Constructs a dataframe of fixed features that are used to initialize the
    hidden state and the LSTM cell.
    '''
    # initialize output dataframe with as-is features
    df = lstgs[BINARY_FEATS + COUNT_FEATS]
    # clock features
    df['start_days'] = lstgs.start_date
    clock = pd.to_datetime(lstgs.start_date, unit='D', origin=START)
    df = df.join(extract_day_feats(clock))
    # slr feedback
    df.loc[df.fdbk_score.isna(), 'fdbk_score'] = 0
    df['fdbk_score'] = df.fdbk_score.astype(np.int64)
    df['fdbk_pstv'] = lstgs['fdbk_pstv'] / 100
    df.loc[df.fdbk_pstv.isna(), 'fdbk_pstv'] = 1
    df['fdbk_100'] = df['fdbk_pstv'] == 1
    # prices
    df['start'] = lstgs['start_price']
    df['decline'] = lstgs['decline_price'] / lstgs['start_price']
    df['accept'] = lstgs['accept_price'] / lstgs['start_price']
    for z in ['start', 'decline', 'accept']:
        df[z + '_round'], df[z +'_nines'] = do_rounding(df[z])
    df['has_decline'] = df['decline'] > 0
    df['has_accept'] = df['accept'] < 1
    df['auto_dist'] = df['accept'] - df['decline']
    # condition
    cndtn = lstgs['cndtn']
    df['new'] = cndtn == 1
    df['used'] = cndtn == 7
    df['refurb'] = cndtn.isin([2, 3, 4, 5, 6])
    df['wear'] = cndtn.isin([8, 9, 10, 11]) * (cndtn - 7)
    # word2vec scores
    df = df.join(get_w2v(lstgs, 'slr'))
    df = df.join(get_w2v(lstgs, 'byr'))
    # time features
    tfcols = [c for c in lstgs.columns if c.startswith('tf_')]
    df = df.join(lstgs[tfcols])
    return df


def do_pca(df):
    df = StandardScaler().fit_transform(df)
    df = PCA(n_components=len(df.columns)).fit_transform(df)
    return df


if __name__ == "__main__":
    # load data
    print('Loading data')
    threads = pd.DataFrame()
    lstgs = pd.DataFrame()
    paths = ['%s/%s' % (CHUNKS_DIR, name) for name in os.listdir(CHUNKS_DIR)
        if os.path.isfile('%s/%s' % (CHUNKS_DIR, name)) and 'frames' in name]
    for path in sorted(paths):
        d = pickle.load(open(path, 'rb'))
        threads = threads.append(d['threads'])
        lstgs = lstgs.append(d['lstgs'])

    # thread features
    print('Creating thread features')
    x_thread = threads[['byr_us', 'byr_hist']]
    pickle.dump(x_thread, open(FRAMES_DIR + 'x_thread.pkl', 'wb'))
    del threads x_thread

    # listing features
    print('Creating listing features')
    x_lstg = get_x_lstg(lstgs)
    x_lstg = do_pca(x_lstg)
    pickle.dump(x_lstg, open(FRAMES_DIR + 'x_lstg.pkl', 'wb'))

    # lookup file
    lookup = lstgs[['start_price', 'decline_price', 'accept_price', 'start_days']]
    pickle.dump(lookup, open(FRAMES_DIR + 'lookup.pkl', 'wb'))