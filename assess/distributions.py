import numpy as np
import pandas as pd
from compress_pickle import load, dump
from processing.processing_utils import load_file
from processing.e_inputs.offer import get_y_msg
from processing.f_discrim.discrim_utils import concat_sim_chunks, get_obs_outcomes
from assess.assess_consts import MAX_THREADS
from processing.processing_consts import INTERVAL
from constants import TEST, PLOT_DIR, BYR_HIST_MODEL, CON_MULTIPLIER, \
    HIST_QUANTILES, SIM, OBS, ARRIVAL_PREFIX, MAX_DELAY, PCTILE_DIR
from featnames import MONTHS_SINCE_LSTG, BYR_HIST, DELAY, EXP, CON, MSG


def get_pdf(y, intervals, add_last=True):
    # construct pdf with integer indices
    v = np.sort((y * intervals).astype('int64').values)
    x, counts = np.unique(v, return_counts=True)
    pdf = counts / len(y)
    pdf = pd.Series(pdf, index=x, name=y.name)
    # reindex to include every interval
    idx = range(intervals+1) if add_last else range(intervals)
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
    pdf = get_pdf(y, MAX_DELAY[1], add_last=False).to_frame()
    pdf['period'] = pdf.index // INTERVAL[1]
    p[ARRIVAL_PREFIX] = pdf.groupby('period').sum().squeeze()

    # buyer history
    y = threads[BYR_HIST] / HIST_QUANTILES
    pdf = get_pdf(y, HIST_QUANTILES, add_last=False)
    pc = load(PCTILE_DIR + '{}.pkl'.format(BYR_HIST))
    pdf.index = get_quantiles(pc, HIST_QUANTILES)
    pdf.drop('NaN', inplace=True)
    p[BYR_HIST_MODEL] = pdf

    return p


def get_distributions(d):
    p = get_arrival_distributions(d['threads'])
    for turn in range(1, 8):
        p[turn] = dict()
        df = d['offers'].xs(turn, level='index')

        # delay
        if turn > 1:
            y = df[DELAY].copy()
            y.loc[df[EXP]] = 1.0  # count expirations and censors together
            p[turn][DELAY] = get_cdf(y, MAX_DELAY[turn])

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
    df = df[~censored]
    s = df.reset_index('index')['index'].groupby(['lstg', 'thread']).count()
    s = s.groupby(s).count() / len(s)
    return s


def main():
    # observed outcomes
    obs = get_obs_outcomes(TEST, drop_censored=False)

    # simulated outcomes
    sim = concat_sim_chunks(TEST, drop_censored=False)

    # loop over models, get observed and simulated distributions
    p = {SIM: get_distributions(sim), OBS: get_distributions(obs)}

    # number of threads per listing
    lstgs = load_file(TEST, 'lookup').index
    p[SIM]['threads'] = num_threads(sim['threads'], lstgs)
    p[OBS]['threads'] = num_threads(obs['threads'], lstgs)

    # number of offers per thread
    p[SIM]['offers'] = num_offers(sim['offers'])
    p[OBS]['offers'] = num_offers(obs['offers'])

    # save
    dump(p, PLOT_DIR + 'distributions.pkl')


if __name__ == '__main__':
    main()
