import numpy as np
import pandas as pd
from compress_pickle import load
from utils import load_file
from constants import PARTS_DIR
from featnames import EXP, DELAY, START_PRICE, LOOKUP


def get_pctiles(s):
    n = len(s.index)
    # create series of index name and values pctile
    idx = pd.Index(np.sort(s.values), name=s.name)
    pctiles = pd.Series(np.arange(1, n+1) / n,
                        index=idx, name='pctile')
    pctiles = pctiles.groupby(pctiles.index).max()
    return pctiles


def discrete_pdf(s, censoring=None):
    s = s.groupby(s).count() / len(s)
    # censor
    if censoring is not None:
        s.loc[censoring] = s[s.index >= censoring].sum(axis=0)
        s = s[s.index <= censoring]
        assert np.abs(s.sum() - 1) < 1e-8
        # relabel index
        idx = s.index.astype(str).tolist()
        idx[-1] += '+'
        s.index = idx
    return s


def load_data(part=None, run_dir=None, relist=True):
    # folder of simulation output
    if run_dir is None:  # using simulated players
        folder = PARTS_DIR + '{}/sim/'.format(part)
    else:
        folder = run_dir + '{}/'.format(part)
    # load dataframes
    threads = load(folder + 'x_thread.gz')
    offers = load(folder + 'x_offer.gz')
    clock = load(folder + 'clock.gz')
    # drop censored offers
    drop = offers[EXP] & (offers[DELAY] < 1)
    offers = offers[~drop]
    clock = clock[~drop]
    assert (clock.xs(1, level='index').index == threads.index).all()
    # start price
    start_price = load_file(part, LOOKUP)[START_PRICE]
    lstgs = threads.index.get_level_values(level='lstg').unique()
    start_price = start_price.reindex(index=lstgs)  # drop infreq arrivals
    # when not relisting, keep only first simulation
    if not relist:
        threads = threads.xs(0, level='sim')
        offers = offers.xs(0, level='sim')
        clock = clock.xs(0, level='sim')
    return threads, offers, clock, start_price
