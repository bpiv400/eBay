import numpy as np
import pandas as pd
from compress_pickle import dump
from assess.assess_utils import get_num_out
from processing.e_inputs.offer import get_y_delay, get_y_con, get_y_msg
from processing.e_inputs.first_arrival import get_first_arrival_period
from processing.e_inputs.next_arrival import get_interarrival_period
from processing.f_discrim.discrim_utils import concat_sim_chunks, get_obs_outcomes
from constants import TEST, MODELS, PLOT_DIR, FIRST_ARRIVAL_MODEL, \
    INTERARRIVAL_MODEL, BYR_HIST_MODEL, OFFER_MODELS
from featnames import BYR_HIST, DELAY, CON, MSG


def get_offer_outcome(m, offers):
    turn = int(m[-1])
    df = offers.xs(turn, level='index')
    if m[:-1] == DELAY:
        y = get_y_delay(df, turn)
    elif m[:-1] == CON:
        y = get_y_con(df)
    elif m[:-1] == MSG:
        y = get_y_msg(df, turn)
    return y


def get_outcomes(start_time, d):
    # collect outcomes in dictionary
    y = dict()
    y[FIRST_ARRIVAL_MODEL] = get_first_arrival_period(start_time,
                                                      d['thread_start'])
    y[INTERARRIVAL_MODEL] = get_interarrival_period(start_time,
                                                    d['thread_start'],
                                                    d['lstg_end'])
    y[BYR_HIST_MODEL] = d['threads'][BYR_HIST]
    for m in OFFER_MODELS:
        y[m] = get_offer_outcome(m, d['offers'])
    return y


def get_distribution(m, y):
    # number of periods
    num_out = get_num_out(m)
    # calculate categorical distribution
    p = np.array([(y == i).mean() for i in range(num_out)])
    # make sure p sums to 1
    assert np.abs(p.sum() - 1) < 1e-8
    return p


def num_threads(df, lstgs):
    s = df.reset_index('thread')['thread'].groupby('lstg').count()
    s = s.reindex(index=lstgs, fill_value=0)
    s = s.groupby(s).count() / len(lstgs)
    return s


def num_offers(df):
    s = df.reset_index('index')['index'].groupby(['lstg', 'thread']).count()
    s = s.groupby(s).count() / len(s)
    return s


def main():
    # observed outcomes
    lookup, obs = get_obs_outcomes(TEST, timestamps=True)

    # simulated outcomes
    sim = concat_sim_chunks(TEST, lookup=lookup)

    # number of threads per listing
    num_threads_obs = num_threads(obs['threads'], lookup.index).rename('obs')
    num_threads_sim = num_threads(sim['threads'], lookup.index).rename('sim')
    df_threads = pd.concat([num_threads_obs, num_threads_sim], axis=1)
    dump(df_threads, PLOT_DIR + 'num_threads.pkl')

    # number of offers per thread
    num_offers_obs = num_offers(obs['offers']).rename('obs')
    num_offers_sim = num_offers(sim['offers']).rename('sim')
    df_offers = pd.concat([num_offers_obs, num_offers_sim], axis=1)
    dump(df_offers, PLOT_DIR + 'num_offers.pkl')

    # loop over models, get observed and simulated distributions
    y_sim = get_outcomes(lookup.start_time, sim)
    y_obs = get_outcomes(lookup.start_time, obs)

    p = dict()
    for m in MODELS:
        p[m] = {'simulated': get_distribution(m, y_sim[m]),
                'observed': get_distribution(m, y_obs[m])}
    dump(p, PLOT_DIR + 'distributions.pkl')


if __name__ == '__main__':
    main()
