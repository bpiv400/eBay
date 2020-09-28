import os
import pandas as pd
from inputs.policy_byr import process_byr_inputs
from inputs.policy_slr import process_slr_inputs
from inputs.util import convert_x_to_numpy
from processing.util import collect_date_clock_feats, get_days_delay, get_norm
from utils import topickle, is_split, load_file
from constants import IDX, DAY, MAX_DAYS, POLICY_BYR, POLICY_SLR
from featnames import DAYS, DELAY, CON, SPLIT, NORM, REJECT, AUTO, EXP, \
    CLOCK_FEATS, TIME_FEATS, OUTCOME_FEATS, DAYS_SINCE_LSTG, \
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
    df[SPLIT] = df[CON].apply(lambda x: is_split(x))
    # total concession
    df[NORM] = get_norm(df[CON])
    # reject auto and exp are last
    df[REJECT] = df[CON] == 0
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR], level='index')
    df[EXP] = df[DELAY] == 1
    # select and sort features
    df = df.loc[:, CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS]
    return df, clock


def process_sim_threads(df=None, lstg_start=None):
    # convert clock to months_since_lstg
    df = df.join(lstg_start)
    df[DAYS_SINCE_LSTG] = (df.clock - df.start_time) / DAY
    assert df[DAYS_SINCE_LSTG].max() < MAX_DAYS
    df = df.drop(['clock', 'start_time'], axis=1)
    # reorder columns to match observed
    thread_cols = [DAYS_SINCE_LSTG, BYR_HIST]
    if 'byr_agent' in df.columns:
        thread_cols += ['byr_agent']
    df = df.loc[:, thread_cols]
    return df


def process_sim_delays(df=None, lstg_start=None):
    df['day'] = (df[CLOCK] - lstg_start.reindex(index=df.index)) // DAY
    df = df.set_index('day', append=True).sort_index()
    return df


def concat_sim_chunks(sims):
    """
    Loops over simulations, concatenates dataframes.
    :param list sims: list of dictionaries of simulated outcomes.
    :return tuple (threads, offers): dataframes of threads and offers.
    """
    # collect chunks
    data = dict()
    for sim in sims:
        for k, v in sim.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    # concatenate
    for k, v in data.items():
        data[k] = pd.concat(v, axis=0).sort_index()
    return data


def process_sims(part=None, sims=None, output_dir=None, byr=None):
    # concatenate chunks
    data = concat_sim_chunks(sims)

    # create output dataframes
    d = dict()
    d[LOOKUP] = load_file(part, LOOKUP)
    d[X_THREAD] = process_sim_threads(df=data['threads'],
                                      lstg_start=d[LOOKUP][START_TIME])
    d[X_OFFER], d[CLOCK] = process_sim_offers(df=data['offers'])

    # agent specific outputs
    if byr is not None:
        if byr:
            d['delays'] = process_sim_delays(df=data['delays'],
                                             lstg_start=d[LOOKUP][START_TIME])
            model_inputs = process_byr_inputs(d)
        else:
            model_inputs = process_slr_inputs(d)
        convert_x_to_numpy(x=model_inputs['x'],
                           idx=model_inputs['y'].index)
        model_inputs['y'] = model_inputs['y'].to_numpy()
        d[POLICY_BYR if byr else POLICY_SLR] = model_inputs

    # create directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # save
    for k, df in d.items():
        if k != LOOKUP:
            topickle(df, output_dir + '{}.pkl'.format(k))
