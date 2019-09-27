import sys
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse, pickle
from datetime import datetime as dt
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from constants import *
from utils import *
from processing_utils import *


def parse_days(diff, t0, t1):
    # count of arrivals by day
    days = diff.dt.days.rename('period')
    days = days[days <= MAX_DAYS].to_frame()
    days = days.assign(count=1)
    days = days.groupby(['lstg', 'period']).sum().squeeze()
    # end of listings
    T1 = int((pd.to_datetime(END) - pd.to_datetime(START)).total_seconds())
    t1.loc[t1[t1 > T1].index] = T1
    end = (pd.to_timedelta(t1 - t0, unit='s').dt.days + 1).rename('period')
    end.loc[end > MAX_DAYS] = MAX_DAYS + 1
    # create multi-index from end stamps
    idx = multiply_indices(end)
    # expand to new index and return
    return days.reindex(index=idx, fill_value=0).sort_index()


def get_y_arrival(lstgs, threads):
    d = {}
    # time_stamps
    t0 = lstgs.start_date * 24 * 3600
    t1 = lstgs.end_time
    diff = pd.to_timedelta(threads.clock - t0, unit='s')
    # append arrivals to end stamps
    d['days'] = parse_days(diff, t0, t1)
    # create other outcomes
    d['loc'] = threads.byr_us.rename('loc')
    d['hist'] = threads.byr_hist.rename('hist')
    d['bin'] = threads.bin
    sec = ((diff.dt.seconds[threads.bin == 0] + 0.5) / (24 * 3600 + 1))
    d['sec'] = sec.rename('sec')
    return d


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
    pca = PCA(n_components=len(df.columns), svd_solver='full')
    df = pca.fit_transform(df)
    return df


if __name__ == "__main__":
    # load dataframes
    print('Loading data')
    lstgs = load_frames('lstgs')
    threads = load_frames('threads')

    # arrival variables
    print('Creating arrival variables')
    y_arrival = get_y_arrival(lstgs, threads)
    pickle.dump(y_arrival, open(FRAMES_DIR + 'y_arrival.pkl', 'wb'))

    # thread features
    print('Creating thread features')
    x_thread = threads[['byr_us', 'byr_hist']]
    pickle.dump(x_thread, open(FRAMES_DIR + 'x_thread.pkl', 'wb'))

    # listing features
    print('Creating listing features')
    x_lstg = get_x_lstg(lstgs)
    x_lstg = do_pca(x_lstg)
    pickle.dump(x_lstg, open(FRAMES_DIR + 'x_lstg.pkl', 'wb'))

    # lookup file
    lookup = lstgs[['start_price', 'decline_price', 'accept_price', 'start_days']]
    pickle.dump(lookup, open(FRAMES_DIR + 'lookup.pkl', 'wb'))