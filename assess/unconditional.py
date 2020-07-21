import numpy as np
import pandas as pd
from compress_pickle import load, dump
from utils import load_file
from inputs.offer import get_y_msg
from inputs.discrim import load_threads_offers
from assess.const import MAX_THREADS
from constants import TEST, PLOT_DIR, BYR_HIST_MODEL, CON_MULTIPLIER, \
    HIST_QUANTILES, SIM, OBS, ARRIVAL, PCTILE_DIR, \
    MAX_DELAY_ARRIVAL, MAX_DELAY_TURN, INTERVAL_ARRIVAL
from featnames import MONTHS_SINCE_LSTG, BYR_HIST, DELAY, EXP, CON, \
    MSG, REJECT


def get_pdf(y, intervals, add_last=True):
    # construct pdf with integer indices
    v = np.sort((y * intervals).astype('int64').values)
    x, counts = np.unique(v, return_counts=True)
    pdf = counts / len(y)
    pdf = pd.Series(pdf, index=x, name=y.name)
    # reindex to include every interval
    idx = range(intervals + 1) if add_last else range(intervals)
    pdf = pdf.reindex(index=idx, fill_value=0.0)
    return pdf


def get_cdf(y, intervals, add_last=True):
    pdf = get_pdf(y, intervals, add_last=add_last)
    return pdf.cumsum()


def get_quantiles(pc, n_quantiles):
    quantiles = np.arange(0, 1, float(1/n_quantiles))
    low = [pc[pc >= d].index[0] for d in quantiles]
    high = [pc[pc < d].index[-1] for d in quantiles[1:]]

    # index of ranges
    idx = []
    for i in range(len(quantiles) - 1):
        if high[i] > low[i]:
            idx.append('{}-{}'.format(low[i], high[i]))
        elif high[i] == low[i]:
            idx.append('{}'.format(low[i]))
        else:
            idx.append('NaN')

    # last entry has no explicit upper bound
    idx.append('{}+'.format(low[-1]))

    return idx


def get_arrival_distributions(threads):
    p = dict()

    # arrival times
    y = threads[MONTHS_SINCE_LSTG]
    pdf = get_pdf(y, MAX_DELAY_ARRIVAL, add_last=False).to_frame()
    pdf['period'] = pdf.index // INTERVAL_ARRIVAL
    p[ARRIVAL] = pdf.groupby('period').sum().squeeze()

    # buyer history
    y = threads[BYR_HIST] / HIST_QUANTILES
    pdf = get_pdf(y, HIST_QUANTILES, add_last=False)
    pc = load(PCTILE_DIR + '{}.pkl'.format(BYR_HIST))
    pdf.index = get_quantiles(pc, HIST_QUANTILES)
    pdf.drop('NaN', inplace=True)
    p[BYR_HIST_MODEL] = pdf

    return p


def get_distributions(threads, offers):
    p = get_arrival_distributions(threads)
    for turn in range(1, 8):
        p[turn] = dict()
        df = offers.xs(turn, level='index')

        # delay
        if turn > 1:
            y = df[DELAY].copy()
            y.loc[df[EXP]] = 1.0  # count expirations and censors together
            p[turn][DELAY] = get_cdf(y, MAX_DELAY_TURN)

        # concession
        p[turn][CON] = get_cdf(df[CON], CON_MULTIPLIER)

        # message
        if turn < 7:
            p[turn][MSG] = get_y_msg(df, turn).mean()

    return p


def num_threads(df, lstgs):
    # threads per listing
    s = df.reset_index('thread')['thread'].groupby('lstg').count()
    s = s.reindex(index=lstgs, fill_value=0)  # fill in zeros
    s = s.groupby(s).count() / len(lstgs)
    # censor at MAX_THREADS
    s.iloc[MAX_THREADS] = s.iloc[MAX_THREADS:].sum(axis=0)
    s = s[:MAX_THREADS + 1]
    idx = s.index.astype(str).tolist()
    idx[-1] += '+'
    s.index = idx
    assert np.abs(s.sum() - 1) < 1e-8
    return s


def num_offers(df):
    censored = df[EXP] & (df[DELAY] < 1)
    byr_reject = df[REJECT] & df.index.isin([3, 5], level='index')
    df = df[~censored & ~byr_reject]
    s = df.reset_index('index')['index'].groupby(['lstg', 'thread']).count()
    s = s.groupby(s).count() / len(s)
    return s


def create_outputs(threads, offers, lstgs):
    # loop over models, get distributions
    d = get_distributions(threads, offers)

    # number of threads per listing
    d['threads'] = num_threads(threads, lstgs)

    # number of offers per thread
    d['offers'] = num_offers(offers)

    return d


def main():
    # lookup file
    lookup = load_file(TEST, 'lookup')
    lstgs = lookup.index

    # observed and simulated outcomes
    threads_obs, offers_obs = load_threads_offers(part=TEST, sim=False)
    threads_sim, offers_sim = load_threads_offers(part=TEST, sim=True)

    # unconditional distributions
    p = dict()
    p[OBS] = create_outputs(threads_obs, offers_obs, lstgs)
    p[SIM] = create_outputs(threads_sim, offers_sim, lstgs)

    # save
    dump(p, PLOT_DIR + 'p.pkl')


if __name__ == '__main__':
    main()
