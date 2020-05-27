import numpy as np
import pandas as pd
from compress_pickle import dump
from processing.processing_utils import read_csv
from processing.processing_consts import CLEAN_DIR
from constants import PCTILE_DIR, START
from featnames import START_PRICE, BYR_HIST


# creates series of percentiles indexed by column variable
def get_pctiles(s):
    N = len(s.index)
    # create series of index name and values pctile
    idx = pd.Index(np.sort(s.values), name=s.name)
    pctiles = pd.Series(np.arange(N) / (N - 1), index=idx, name='pctile')
    pctiles = pctiles.groupby(pctiles.index).min()
    # put max in 99th percentile
    pctiles.loc[pctiles == 1] -= 1e-16
    # reformat series with index s.index and values pctiles
    s = s.to_frame().join(pctiles, on=s.name).drop(
        s.name, axis=1).squeeze().rename(s.name)
    return s, pctiles


def get_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second


def main():
    # load files
    offers = read_csv('offers')
    threads = read_csv('threads')
    listings = read_csv('listings')

    # convert byr_hist to pctiles and save threads
    threads.loc[:, BYR_HIST], toSave = \
        get_pctiles(threads[BYR_HIST])
    dump(toSave, PCTILE_DIR + '{}.pkl'.format(BYR_HIST))
    dump(threads, CLEAN_DIR + 'threads.pkl')

    # arrival time
    thread_start = offers.clock.xs(1, level='index')
    s = pd.to_datetime(thread_start, origin=START, unit='s')

    # split off missing
    toReplace = threads.bin & ((thread_start + 1) % (24 * 3600) == 0)
    missing = pd.to_datetime(s.loc[toReplace].dt.date)
    labeled = s.loc[~toReplace]

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
    last = last.loc[~toReplace]
    last = last.groupby('lstg').max()
    last = pd.to_datetime(last, origin=START, unit='s')
    last = last.reindex(index=missing.index, level='lstg').dropna()

    # restrict censoring seconds to same day as bin with missing time
    sameDate = pd.to_datetime(last.dt.date) == missing.reindex(last.index)
    lower = get_seconds(last.dt).loc[sameDate].rename('lower')

    # make end time one second after last observed same day offer before BIN
    lower += 1
    lower.loc[lower == 3600 * 24] -= 1

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
    dump(offers, CLEAN_DIR + 'offers.pkl')

    # update listing end time
    listings.loc[end_time.index, 'end_time'] = end_time

    # add arrivals per day to listings
    arrivals = thread_start.groupby('lstg').count().reindex(
        index=listings.index, fill_value=0)
    duration = (listings.end_time + 1) / (24 * 3600) - listings.start_date
    listings['arrival_rate'] = arrivals / duration

    # add start_price percentile
    listings['start_price_pctile'], _ = \
        get_pctiles(listings[START_PRICE])

    # convert fdbk_pstv to a rate
    listings.loc[listings.fdbk_score < 0, 'fdbk_score'] = 0
    listings.loc[:, 'fdbk_pstv'] = listings.fdbk_pstv / listings.fdbk_score
    listings.loc[listings.fdbk_pstv.isna(), 'fdbk_pstv'] = 1

    # replace count and rate variables with percentiles
    for feat in ['fdbk_score', 'photos', 'slr_lstg_ct', 'slr_bo_ct', 'arrival_rate']:
        print(feat)
        listings.loc[:, feat], toSave = get_pctiles(listings[feat])
        if feat != 'arrival_rate':
            dump(toSave, PCTILE_DIR + '{}.pkl'.format(feat))

    # save listings
    dump(listings, CLEAN_DIR + 'listings.pkl')


if __name__ == '__main__':
    main()
