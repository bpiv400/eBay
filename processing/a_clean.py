import os
import numpy as np
import pandas as pd
from processing.util import get_con, get_norm, get_pctiles
from utils import topickle
from constants import CLEAN_DIR, FEATS_DIR, PARTS_DIR, PCTILE_DIR, START, SEED, \
    SHARES, NUM_CHUNKS, DAY
from featnames import START_PRICE, BYR_HIST, NORM, SLR

# data types for csv read
OTYPES = {'lstg': 'int64',
          'thread': 'int64',
          'index': 'uint8',
          'clock': 'int64',
          'price': 'float64',
          'accept': bool,
          'reject': bool,
          'censored': bool,
          'message': bool}

TTYPES = {'lstg': 'int64',
          'thread': 'int64',
          'byr': 'int64',
          'byr_hist': 'int64',
          'bin': bool,
          'byr_us': bool}

LTYPES = {'lstg': 'int64',
          'slr': 'int64',
          'meta': 'int64',
          'leaf': 'int64',
          'cndtn': 'uint8',
          'start_date': 'uint16',
          'end_time': 'int64',
          'fdbk_score': 'int64',
          'fdbk_pstv': 'int64',
          'start_price': 'float64',
          'photos': 'uint8',
          'slr_lstg_ct': 'int64',
          'slr_bo_ct': 'int64',
          'decline_price': 'float64',
          'accept_price': 'float64',
          'store': bool,
          'slr_us': bool,
          'fast': bool}

DATA_TYPES = {'listings': LTYPES,
              'threads': TTYPES,
              'offers': OTYPES}

# indices when reading in CSVs
IDX_NAMES = {'offers': ['lstg', 'thread', 'index'],
             'threads': ['lstg', 'thread'],
             'listings': 'lstg'}


def create_chunks(listings=None, threads=None, offers=None):
    """
    Chunks data by listing.
    :param DataFrame listings: listing features with index ['lstg']
    :param DataFrame threads: thread features with index ['lstg', 'thread']
    :param DataFrame offers: offer features with index ['lstg', 'thread', 'index']
    """
    # output directory
    chunk_dir = FEATS_DIR + 'chunks/'
    if not os.path.isdir(chunk_dir):
        os.mkdir(chunk_dir)
    # add norm to offers
    con = get_con(offers.price.unstack(), listings[START_PRICE])
    offers[NORM] = get_norm(con)
    # split into chunks by seller
    slrs = listings[SLR].reset_index().sort_values(
        by=[SLR, 'lstg']).set_index(SLR).squeeze()
    u = np.unique(listings[SLR].values)
    groups = np.array_split(u, NUM_CHUNKS)
    for i in range(NUM_CHUNKS):
        print('Creating chunk {} of {}'.format(i + 1, NUM_CHUNKS))
        lstgs = slrs.loc[groups[i]].values
        chunk = {'listings': listings.reindex(index=lstgs),
                 'threads': threads.reindex(index=lstgs, level='lstg'),
                 'offers': offers.reindex(index=lstgs, level='lstg')}
        topickle(chunk, chunk_dir + '{}.pkl'.format(i))


def partition_lstgs(s):
    # series of index slr and value lstg
    slrs = s.reset_index().sort_values(
        by=['slr', 'lstg']).set_index('slr').squeeze()
    # randomly order sellers
    u = np.unique(slrs.index.values)
    np.random.seed(SEED)   # set seed
    np.random.shuffle(u)
    # partition listings into dictionary
    d = {}
    last = 0
    for key, val in SHARES.items():
        curr = last + int(u.size * val)
        d[key] = np.sort(slrs.loc[u[last:curr]].values)
        last = curr
    d['testing'] = np.sort(slrs.loc[u[last:]].values)
    return d


def read_csv(name):
    """
    Reads in one of three csvs of features.
    :param str name: one of ['listings', 'threads', 'offers']
    :return: datafrome of features
    """
    filename = CLEAN_DIR + '{}.csv'.format(name)
    df = pd.read_csv(filename, dtype=DATA_TYPES[name])
    df = df.set_index(IDX_NAMES[name])
    return df


def get_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second


def series_to_pctiles(s=None, pctiles=None):
    """
    Converts values into percentiles
    :param pandas.Series s: values from which percentiles are calculated from
    :param pandas.Series pctiles: corresponding percentiles
    :return: pandas.Series with same index as s and values from pctiles
    """
    if pctiles is None:
        pctiles = get_pctiles(s)
    return s.to_frame().join(pctiles, on=s.name).drop(
        s.name, axis=1).squeeze().rename(s.name)


def main():
    # load files
    offers = read_csv('offers')
    threads = read_csv('threads')
    listings = read_csv('listings')

    # convert byr_hist to pctiles and save threads
    to_save = get_pctiles(threads[BYR_HIST])
    topickle(to_save, PCTILE_DIR + '{}.pkl'.format(BYR_HIST))
    topickle(threads, FEATS_DIR + 'threads.pkl')

    # arrival time
    thread_start = offers.clock.xs(1, level='index')
    s = pd.to_datetime(thread_start, origin=START, unit='s')

    # split off missing
    to_replace = threads.bin & ((thread_start + 1) % DAY == 0)
    missing = pd.to_datetime(s.loc[to_replace].dt.date)
    labeled = s.loc[~to_replace]

    # cdf using labeled data
    N = len(labeled.index)
    sec = pd.Series(np.arange(1, N + 1) / N,
                    index=np.sort(get_seconds(labeled.dt).values),
                    name='pctile')
    pctile = sec.groupby(sec.index).min()
    pctile.loc[-1] = 0
    pctile = pctile.sort_index().rename('pctile')
    cdf = pctile.rename('x').reset_index().set_index('x').squeeze()

    # calculate censoring second
    last = offers.loc[~offers.censored, 'clock'].groupby(
        ['lstg', 'thread']).max()
    last = last.loc[~to_replace]
    last = last.groupby('lstg').max()
    last = pd.to_datetime(last, origin=START, unit='s')
    last = last.reindex(index=missing.index, level='lstg').dropna()

    # restrict censoring seconds to same day as bin with missing time
    same_date = pd.to_datetime(last.dt.date) == missing.reindex(last.index)
    lower = get_seconds(last.dt).loc[same_date].rename('lower')

    # make end time one second after last observed same day offer before BIN
    lower += 1
    lower.loc[lower == DAY] -= 1

    # uniform random
    rand = pd.Series(np.random.rand(len(missing.index)),
                     index=missing.index, name='x')

    # amend rand for censored observations
    tau = lower.to_frame().join(pctile, on='lower')['pctile']
    rand.loc[tau.index] = tau

    # read off of cdf
    newsec = cdf.reindex(index=rand, method='ffill').values
    delta = pd.Series(pd.to_timedelta(newsec, unit='second'),
                      index=rand.index)

    # new bin arrival times
    tdiff = missing + delta - pd.to_datetime(START)
    tdiff = tdiff.dt.total_seconds().astype('int64')

    # end time of listing
    end_time = tdiff.reset_index(
        'thread', drop=True).rename('end_time')

    # update offers clock
    df = offers.clock.reindex(index=end_time.index, level='lstg')
    df = df.to_frame().join(end_time)
    idx = df[df['clock'] > df['end_time']].index
    offers.loc[idx, 'clock'] = df.loc[idx, 'end_time']

    # save offers and threads
    topickle(offers, FEATS_DIR + 'offers.pkl')

    # update listing end time
    listings.loc[end_time.index, 'end_time'] = end_time

    # add arrivals per day to listings
    arrivals = thread_start.groupby('lstg').count().reindex(
        index=listings.index, fill_value=0)
    duration = (listings.end_time + 1) / DAY - listings.start_date
    listings['arrival_rate'] = arrivals / duration

    # add start_price percentile
    listings['start_price_pctile'] = series_to_pctiles(s=listings[START_PRICE])

    # convert fdbk_pstv to a rate
    listings.loc[listings.fdbk_score < 0, 'fdbk_score'] = 0
    listings.loc[:, 'fdbk_pstv'] = listings.fdbk_pstv / listings.fdbk_score
    listings.loc[listings.fdbk_pstv.isna(), 'fdbk_pstv'] = 1

    # replace count and rate variables with percentiles
    listings.loc[:, 'arrival_rate'] = series_to_pctiles(s=listings['arrival_rate'])
    for feat in ['fdbk_score', 'photos', 'slr_lstg_ct', 'slr_bo_ct']:
        print(feat)
        pctiles = get_pctiles(listings[feat])
        listings.loc[:, feat] = series_to_pctiles(
            s=listings[feat], pctiles=pctiles)
        topickle(pctiles, PCTILE_DIR + '{}.pkl'.format(feat))

    # save listings
    topickle(listings, FEATS_DIR + 'listings.pkl')

    # chunk by listing
    create_chunks(listings=listings, threads=threads, offers=offers)

    # partition by seller
    partitions = partition_lstgs(listings[SLR])
    topickle(partitions, PARTS_DIR + 'partitions.pkl')


if __name__ == '__main__':
    main()
