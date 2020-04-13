import pandas as pd
from compress_pickle import dump
from processing.processing_utils import load_file
from processing.f_discrim.discrim_utils import concat_sim_chunks, get_obs_outcomes
from constants import TEST, PLOT_DIR, OFFER_MODELS, DELAY_MODELS
from featnames import MONTHS_SINCE_LSTG, BYR_HIST, DELAY, EXP


def get_outcomes(d):
    # collect outcomes in dictionary
    y = dict()
    # arrival models
    y['arrivals'] = d['threads'][MONTHS_SINCE_LSTG]
    y[BYR_HIST] = d['threads'][BYR_HIST]
    # select offer component for offer models
    for m in OFFER_MODELS:
        outcome, turn = m[:-1], int(m[-1])
        y[m] = d['offers'].loc[:, outcome].xs(turn, level='index')
        # for delay models, count expirations and censors together
        if m in DELAY_MODELS:
            exp = d['offers'].loc[:, EXP].xs(turn, level='index')
            y[m][exp] = 1.0
    return y


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
    y = {'simulated': get_outcomes(sim),
         'observed': get_outcomes(obs)}

    # save
    dump(y, PLOT_DIR + 'outcomes.pkl')


if __name__ == '__main__':
    main()
