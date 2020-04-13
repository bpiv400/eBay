import numpy as np
import pandas as pd
from compress_pickle import dump
from assess.assess_utils import get_num_out
from processing.processing_utils import load_file
from processing.f_discrim.discrim_utils import concat_sim_chunks, get_obs_outcomes
from constants import TEST, PLOT_DIR, OFFER_MODELS, DELAY_MODELS, \
    CON_MODELS, CON_MULTIPLIER
from featnames import MONTHS_SINCE_LSTG, BYR_HIST, DELAY, EXP


def p_multinomial(m, y):
    # number of periods
    num_out = get_num_out(m)
    # calculate categorical distribution
    x = range(num_out)
    p = np.array([(y == i).mean() for i in x])
    pdf = pd.Series(p, index=x, name=y.name)
    # make sure pdf sums to 1
    assert np.abs(pdf.sum() - 1) < 1e-8
    return pdf


def get_cdf(y):
    v = np.sort(y.values)
    x, counts = np.unique(v, return_counts=True)
    pdf = counts / len(v)
    cdf = pd.Series(np.cumsum(pdf), index=x, name=y.name)
    return cdf


def get_distributions(d):
    p = dict()
    # arrival times
    y = d['threads'][MONTHS_SINCE_LSTG]
    p['arrival'] = get_cdf(y)
    # buyer history
    y = d['threads'][BYR_HIST]
    p[BYR_HIST] = p_multinomial(BYR_HIST, y)
    # offer models
    for m in OFFER_MODELS:
        outcome, turn = m[:-1], int(m[-1])
        y = d['offers'].loc[:, outcome].xs(turn, level='index')
        # use integer concessions for con models
        if m in CON_MODELS:
            y = (y * CON_MULTIPLIER).astype('uint8')
        # for delay models, count expirations and censors together
        if m in DELAY_MODELS:
            exp = d['offers'].loc[:, EXP].xs(turn, level='index')
            y[exp] = 1.0
        # convert to distribution
        if y.dtype == 'float64':
            p[m] = get_cdf(y)
        else:
            p[m] = p_multinomial(m, y)
    return p


def num_threads(df, lstgs):
    s = df.reset_index('thread')['thread'].groupby('lstg').count()
    s = s.reindex(index=lstgs, fill_value=0)
    s = s.groupby(s).count() / len(lstgs)
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

    # number of threads per listing
    lstgs = load_file(TEST, 'lookup').index
    num_threads_obs = num_threads(obs['threads'], lstgs)
    num_threads_sim = num_threads(sim['threads'], lstgs)
    df_threads = pd.concat([num_threads_obs.rename('observed'),
                            num_threads_sim.rename('simulated')], axis=1)
    dump(df_threads, PLOT_DIR + 'num_threads.pkl')

    # number of offers per thread
    num_offers_obs = num_offers(obs['offers'])
    num_offers_sim = num_offers(sim['offers'])
    df_offers = pd.concat([num_offers_obs.rename('observed'),
                           num_offers_sim.rename('simulated')], axis=1)
    dump(df_offers, PLOT_DIR + 'num_offers.pkl')

    # loop over models, get observed and simulated distributions
    p = {'simulated': get_distributions(sim),
         'observed': get_distributions(obs)}

    # save
    dump(p, PLOT_DIR + 'distributions.pkl')


if __name__ == '__main__':
    main()
