import os
import numpy as np
import pandas as pd
from processing.util import get_norm
from utils import topickle, feat_to_pctile
from constants import CLEAN_DIR, FEATS_DIR, PARTS_DIR, PCTILE_DIR, START, SEED, \
    SHARES, NUM_CHUNKS, DAY
from featnames import START_PRICE, BYR_HIST, NORM, BYR, SLR, LSTG, THREAD, INDEX, \
    CLOCK, CON, ACCEPT, REJECT, META, LEAF, SLR_BO_CT, START_DATE, END_TIME, \
    STORE, ACC_PRICE, DEC_PRICE

# data types for csv read
OTYPES = {LSTG: 'int64',
          THREAD: 'int64',
          INDEX: 'uint8',
          CLOCK: 'int64',
          'price': 'float64',
          'accept': bool,
          REJECT: bool,
          'message': bool}

TTYPES = {LSTG: 'int64',
          THREAD: 'int64',
          BYR: 'int64',
          BYR_HIST: 'int64',
          'bin': bool,
          'byr_us': bool}

LTYPES = {LSTG: 'int64',
          SLR: 'int64',
          META: 'int64',
          LEAF: 'int64',
          'cndtn': 'uint8',
          START_DATE: 'uint16',
          END_TIME: 'int64',
          'fdbk_score': 'int64',
          'fdbk_pstv': 'int64',
          START_PRICE: 'float64',
          'photos': 'uint8',
          'slr_lstg_ct': 'int64',
          SLR_BO_CT: 'int64',
          DEC_PRICE: 'float64',
          ACC_PRICE: 'float64',
          STORE: bool,
          'slr_us': bool,
          'fast': bool}

DATA_TYPES = {'listings': LTYPES,
              'threads': TTYPES,
              'offers': OTYPES}

# indices when reading in CSVs
IDX_NAMES = {'offers': [LSTG, THREAD, INDEX],
             'threads': [LSTG, THREAD],
             'listings': LSTG}


def create_slr_chunks(listings=None, threads=None, offers=None, chunk_dir=None):
    """
    Chunks data by listing.
    :param DataFrame listings: listing features with index ['lstg']
    :param DataFrame threads: thread features with index ['lstg', 'thread']
    :param DataFrame offers: offer features with index ['lstg', 'thread', 'index']
    :param str chunk_dir: path to output directory
    """
    # split into chunks by seller
    slr = listings[SLR].reset_index().sort_values(
        by=[SLR, LSTG]).set_index(SLR).squeeze()
    u = np.unique(slr.index)
    groups = np.array_split(u, NUM_CHUNKS)
    for i in range(NUM_CHUNKS):
        print('Creating slr chunk {} of {}'.format(i + 1, NUM_CHUNKS))
        lstgs = slr.loc[groups[i]].values
        chunk = {'listings': listings.reindex(index=lstgs),
                 'threads': threads.reindex(index=lstgs, level=LSTG),
                 'offers': offers.reindex(index=lstgs, level=LSTG)}
        topickle(chunk, chunk_dir + '{}.pkl'.format(i))


def create_meta_chunks(listings=None, threads=None, offers=None, chunk_dir=None):
    """
    Chunks data by listing.
    :param DataFrame listings: listing features with index ['lstg']
    :param DataFrame threads: thread features with index ['lstg', 'thread']
    :param DataFrame offers: offer features with index ['lstg', 'thread', 'index']
    :param str chunk_dir: path to output directory
    """
    # split into chunks by seller
    meta = listings[META].reset_index().sort_values(
        by=[META, LSTG]).set_index(META).squeeze()
    u = np.unique(meta.index)
    for i in range(len(u)):
        print('Creating meta chunk {} of {}'.format(i + 1, len(u)))
        lstgs = meta.loc[u[i]].values
        chunk = {'listings': listings.reindex(index=lstgs),
                 'threads': threads.reindex(index=lstgs, level=LSTG),
                 'offers': offers.reindex(index=lstgs, level=LSTG)}
        topickle(chunk, chunk_dir + 'meta{}.pkl'.format(i))


def partition_lstgs(s):
    # series of index slr and value lstg
    slrs = s.reset_index().sort_values(
        by=[SLR, LSTG]).set_index(SLR).squeeze()
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
    d['test'] = np.sort(slrs.loc[u[last:]].values)
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


def get_pctiles(s=None):
    """
    Converts values into percentiles.
    :param s: values to calculate percentiles from
    :return: pandas.Series with index of unique values and percentiles as values
    """
    rates = s.groupby(s).count() / len(s)
    return rates.cumsum().shift(fill_value=0.).rename('pctile')


def get_con(price=None, start_price=None):
    # unstack
    price = price.unstack()
    # compute concessions
    con = pd.DataFrame(index=price.index)
    con[1] = price[1] / start_price
    con[2] = (price[2] - start_price) / (price[1] - start_price)
    for i in range(3, 8):
        con[i] = (price[i] - price[i-2]) / (price[i-1] - price[i-2])
    # stack into series
    con = con.rename_axis('index', axis=1).stack()
    # first buyer concession should be greater than 0
    assert con.loc[con.index.isin([1], level=INDEX)].min() > 0
    # round concessions
    rounded = np.round(con, decimals=2)
    rounded.loc[(rounded == 1) & (con < 1)] = 0.99
    rounded.loc[(rounded == 0) & (con > 0)] = 0.01
    return rounded


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
    thread_start = offers[CLOCK].xs(1, level=INDEX)
    ts = pd.to_datetime(thread_start, origin=START, unit='s')

    # split off missing
    to_replace = threads.bin & ((thread_start + 1) % DAY == 0)
    missing = pd.to_datetime(ts.loc[to_replace].dt.date)

    # calculate censoring second
    last = offers[CLOCK].groupby([LSTG, THREAD]).max()
    last = last.loc[~to_replace]
    last = last.groupby(LSTG).max()
    last = pd.to_datetime(last, origin=START, unit='s')
    last = last.reindex(index=missing.index, level=LSTG).dropna()

    # restrict censoring seconds to same day as bin with missing time
    same_date = pd.to_datetime(last.dt.date) == missing.reindex(last.index)
    lower = get_seconds(last.dt).loc[same_date].rename('lower')

    # uniform random
    np.random.seed(SEED)
    rand = pd.Series(np.random.rand(len(missing.index)),
                     index=missing.index, name='x')

    # amend rand for censored observations
    sec = thread_start[~to_replace] % DAY
    pdf = sec.groupby(sec).count() / len(sec)
    pdf = pdf.reindex(index=range(DAY), fill_value=0)
    pctile = pdf.cumsum().rename('pctile')
    tau = lower.to_frame().join(pctile, on='lower')['pctile']

    rand.loc[tau.index] *= 1 - tau
    rand.loc[tau.index] += tau

    # read off of cdf
    cdf = pctile.rename('x').reset_index().set_index('x').squeeze()
    assert not cdf.index.duplicated().max()
    newsec = cdf.reindex(index=rand, method='pad').values
    newsec[np.isnan(newsec)] = 0
    assert newsec.min() >= 0
    assert newsec.max() < DAY
    delta = pd.Series(pd.to_timedelta(newsec, unit='second'),
                      index=rand.index)

    # new bin arrival times
    tdiff = missing + delta - pd.to_datetime(START)
    tdiff = tdiff.dt.total_seconds().astype('int64')

    # end time of listing
    end_time = tdiff.reset_index(THREAD, drop=True).rename('end_time')

    # update offers clock
    df = offers[CLOCK].reindex(index=end_time.index, level=LSTG)
    df = df.to_frame().join(end_time)
    idx = df[df[CLOCK] > df['end_time']].index
    offers.loc[idx, CLOCK] = df.loc[idx, 'end_time']

    # update listing end time
    listings.loc[end_time.index, 'end_time'] = end_time

    # add concession and norm to offers and save
    offers[CON] = get_con(offers['price'], listings[START_PRICE])
    offers[NORM] = get_norm(offers[CON])

    assert offers.loc[offers[CON] == 1, ACCEPT].all()
    assert offers.loc[offers[CON] == 0, REJECT].all()
    assert (offers.loc[offers[ACCEPT], CON] == 1).all()
    assert (offers.loc[offers[REJECT], CON] == 0).all()

    topickle(offers, FEATS_DIR + 'offers.pkl')

    # add arrivals per day to listings, as percentile
    arrivals = thread_start.groupby(LSTG).count().reindex(
        index=listings.index, fill_value=0)
    duration = (listings.end_time + 1) / DAY - listings.start_date
    arrival_rate = arrivals / duration
    listings['arrival_rate'] = feat_to_pctile(s=arrival_rate,
                                              pc=get_pctiles(arrival_rate))

    # convert fdbk_pstv to a rate
    listings.loc[listings.fdbk_score < 0, 'fdbk_score'] = 0
    listings.loc[:, 'fdbk_pstv'] = listings.fdbk_pstv / listings.fdbk_score
    listings.loc[listings.fdbk_pstv.isna(), 'fdbk_pstv'] = 1

    # save percentiles for count features
    for feat in ['fdbk_score', 'photos', 'slr_lstg_ct', SLR_BO_CT, START_PRICE]:
        print(feat)
        pctiles = get_pctiles(listings[feat])
        topickle(pctiles, PCTILE_DIR + '{}.pkl'.format(feat))

    # save listings
    topickle(listings, FEATS_DIR + 'listings.pkl')

    # chunk by listing
    chunk_dir = FEATS_DIR + 'chunks/'
    if not os.path.isdir(chunk_dir):
        os.mkdir(chunk_dir)
    kw = dict(listings=listings, threads=threads, offers=offers, chunk_dir=chunk_dir)
    create_slr_chunks(**kw)
    create_meta_chunks(**kw)

    # partition by seller
    partitions = partition_lstgs(listings[SLR])
    topickle(partitions, PARTS_DIR + 'partitions.pkl')


if __name__ == '__main__':
    main()
