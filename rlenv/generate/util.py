import os
import pandas as pd
from inputs.agents import process_inputs
from inputs.util import convert_x_to_numpy
from processing.util import collect_date_clock_feats, get_days_delay, get_norm, \
    get_common_cons
from utils import topickle, load_file, get_role
from constants import IDX, DAY, MAX_DAYS
from featnames import DAYS, DELAY, CON, COMMON, NORM, REJECT, AUTO, EXP, \
    CLOCK_FEATS, TIME_FEATS, OUTCOME_FEATS, DAYS_SINCE_LSTG, INDEX, \
    BYR_HIST, START_TIME, LOOKUP, SLR, X_THREAD, X_OFFER, CLOCK


def diff_tf(df):
    # pull out time feats
    tf = df[TIME_FEATS].astype('float64')
    df.drop(TIME_FEATS, axis=1, inplace=True)
    # for each feat, unstack, difference, and stack
    for c in TIME_FEATS:
        wide = tf[c].unstack()
        first = wide[[1]].stack()
        diff = wide.diff(axis=1).stack()
        df[c] = pd.concat([first, diff], axis=0).sort_index()
        assert df[c].isna().sum() == 0
    return df


def process_sim_offers(df=None):
    # difference time features
    df = diff_tf(df)
    # clock features
    clock = df[CLOCK]
    df.drop(CLOCK, axis=1, inplace=True)
    df = df.join(collect_date_clock_feats(clock))
    # days and delay
    df[DAYS], df[DELAY] = get_days_delay(clock.unstack())
    # concession as a decimal
    df.loc[:, CON] /= 100
    # indicator for split
    df[COMMON] = get_common_cons(df[CON])
    # total concession
    df[NORM] = get_norm(df[CON])
    # reject auto and exp are last
    df[REJECT] = df[CON] == 0
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR], level=INDEX)
    df[EXP] = df[DELAY] == 1
    # select and sort features
    df = df.loc[:, CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS]
    return df, clock


def process_sim_threads(df=None, lstg_start=None):
    # convert clock to months_since_lstg
    df = df.join(lstg_start)
    df[DAYS_SINCE_LSTG] = (df[CLOCK] - df[START_TIME]) / DAY
    assert df[DAYS_SINCE_LSTG].max() < MAX_DAYS
    df = df.drop([CLOCK, START_TIME], axis=1)
    # reorder columns to match observed
    thread_cols = [DAYS_SINCE_LSTG, BYR_HIST]
    df = df.loc[:, thread_cols]
    return df


def concat_sim_chunks(sims):
    """
    Loops over simulations, concatenates dataframes.
    :param list sims: list of dictionaries of simulated outcomes.
    :return tuple (threads, offers): dataframes of threads and offers.
    """
    # collect chunks
    data = {}
    for sim in sims:
        for k, v in sim.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    # concatenate
    for k, v in data.items():
        data[k] = pd.concat(v, axis=0).sort_index()
    return data


def process_sims(part=None, sims=None, output_dir=None,
                 byr=None, save_inputs=True):
    # concatenate chunks
    data = concat_sim_chunks(sims)

    # create output dataframes
    d = dict()
    d[LOOKUP] = load_file(part, LOOKUP)
    d[X_THREAD] = process_sim_threads(df=data[X_THREAD],
                                      lstg_start=d[LOOKUP][START_TIME])
    d[X_OFFER], d[CLOCK] = process_sim_offers(df=data[X_OFFER])

    # create directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # save
    for k, df in d.items():
        if k != LOOKUP:
            topickle(df, output_dir + '{}.pkl'.format(k))

    # model inputs for agent logs
    if byr is not None and save_inputs:
        inputs = process_inputs(data=d, byr=byr, agent=True)
        convert_x_to_numpy(x=inputs['x'], idx=inputs['y'].index)
        inputs['y'] = inputs['y'].to_numpy()
        topickle(inputs, output_dir + '{}.pkl'.format(get_role(byr)))