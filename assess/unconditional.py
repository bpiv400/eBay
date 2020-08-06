import numpy as np
import pandas as pd
from compress_pickle import dump
from inputs.offer import get_y_msg
from inputs.discrim import load_threads_offers
from processing.util import hist_to_pctile
from assess.util import discrete_pdf
from utils import load_file
from constants import TEST, PLOT_DIR, BYR_HIST_MODEL, CON_MULTIPLIER, \
    MAX_DELAY_ARRIVAL, MAX_DELAY_TURN, INTERVAL_ARRIVAL
from featnames import MONTHS_SINCE_LSTG, BYR_HIST, DELAY, CON, MSG, \
    SIM, OBS, ARRIVAL


def get_pdf(y=None, intervals=None, add_last=False):
    # construct pdf with integer indices
    v = np.sort(y.values)
    x, counts = np.unique(v, return_counts=True)
    pdf = pd.Series(counts / len(y), index=x, name=y.name)
    # reindex to include every interval
    idx = range(intervals + 1) if add_last else range(intervals)
    pdf = pdf.reindex(index=idx, fill_value=0.0)
    return pdf


def get_cdf(y=None, intervals=None):
    pdf = get_pdf(y, intervals, add_last=True)
    return pdf.cumsum()


def get_arrival_distributions(threads):
    p = dict()

    # arrival times
    y = (threads[MONTHS_SINCE_LSTG] * MAX_DELAY_ARRIVAL).astype('int64')
    pdf = get_pdf(y=y, intervals=MAX_DELAY_ARRIVAL).to_frame()
    pdf['period'] = np.array(pdf.index) // INTERVAL_ARRIVAL
    p[ARRIVAL] = pdf.groupby('period').sum().squeeze()

    # buyer history
    y = hist_to_pctile(threads[BYR_HIST], reverse=True)
    p[BYR_HIST_MODEL] = get_cdf(y=y, intervals=y.max())

    return p


def get_distributions(threads=None, offers=None):
    p = get_arrival_distributions(threads)
    for turn in range(1, 8):
        p[turn] = dict()
        df = offers.xs(turn, level='index')

        # delay (no censored delays in data)
        if turn > 1:
            p[turn][DELAY] = get_cdf(y=df[DELAY] * MAX_DELAY_TURN,
                                     intervals=MAX_DELAY_TURN)

        # concession
        p[turn][CON] = get_cdf(y=df[CON] * CON_MULTIPLIER,
                               intervals=CON_MULTIPLIER)

        # message
        if turn < 7:
            p[turn][MSG] = get_y_msg(df, turn).mean()

    return p


def num_threads(threads=None, lstgs=None):
    s = threads.reset_index('thread')['thread'].groupby('lstg').count()
    s = s.reindex(index=lstgs, fill_value=0)  # listings w/o threads
    return discrete_pdf(s, censoring=3)


def num_offers(offers=None):
    s = offers.iloc[:, 0].groupby(['lstg', 'thread']).count()
    return discrete_pdf(s)


def create_outputs(threads=None, offers=None, lstgs=None):
    # loop over models, get distributions
    d = get_distributions(threads=threads, offers=offers)

    # number of threads per listing
    d['threads'] = num_threads(lstgs=lstgs, threads=threads)

    # number of offers per thread
    d['offers'] = num_offers(offers=offers)

    return d


def main():
    # observed and simulated outcomes
    threads_obs, offers_obs = load_threads_offers(part=TEST, sim=False)
    threads_sim, offers_sim = load_threads_offers(part=TEST, sim=True)

    # lookup file
    lstgs = load_file(TEST, 'sim/x_thread').index.get_level_values(
        level='lstg').unique()

    # unconditional distributions
    p = dict()
    p[OBS] = create_outputs(threads=threads_obs,
                            offers=offers_obs,
                            lstgs=lstgs)
    p[SIM] = create_outputs(threads=threads_sim,
                            offers=offers_sim,
                            lstgs=lstgs)

    # save
    dump(p, PLOT_DIR + 'p.pkl')


if __name__ == '__main__':
    main()
