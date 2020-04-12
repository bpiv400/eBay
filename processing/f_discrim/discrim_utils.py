import numpy as np
import pandas as pd
from compress_pickle import load, dump
from processing.e_inputs.inputs_utils import save_sizes, convert_x_to_numpy, save_small
from processing.processing_utils import load_file, collect_date_clock_feats, \
    get_days_delay, get_norm
from utils import is_split
from processing.processing_consts import CLEAN_DIR
from constants import TRAIN_RL, VALIDATION, INPUT_DIR, SIM_CHUNKS, ENV_SIM_DIR, \
    IDX, SLR_PREFIX, MONTH
from featnames import DAYS, DELAY, CON, SPLIT, NORM, REJECT, AUTO, EXP, CENSORED, \
    CLOCK_FEATS, TIME_FEATS, OUTCOME_FEATS, MONTHS_SINCE_LSTG, BYR_HIST


def process_sim_offers(df, lstg_end, keep_tf=True):
    # censor timestamps
    clock = df.clock
    df = df.drop('clock', axis=1)
    clock = np.minimum(clock, lstg_end.reindex(index=df.index, level='lstg'))
    # clock features
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
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR_PREFIX], level='index')
    df[EXP] = (df[DELAY] == 1) | df[CENSORED]
    # difference time feats
    if keep_tf:
        tf = df[TIME_FEATS].astype('float64')
        df.drop(TIME_FEATS, axis=1, inplace=True)
        for c in TIME_FEATS:
            wide = tf[c].unstack()
            first = wide[[1]].stack()
            diff = wide.diff(axis=1).stack()
            df[c] = pd.concat([first, diff], axis=0).sort_index()
            assert df[c].isna().sum() == 0
        df = df.loc[:, CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS]
    else:
        df = df.loc[:, CLOCK_FEATS + OUTCOME_FEATS]
    return df, clock


def process_sim_threads(df, start_time):
    # convert clock to months_since_lstg
    df = df.join(start_time)
    df[MONTHS_SINCE_LSTG] = (df.clock - df.start_time) / MONTH
    df = df.drop(['clock', 'start_time'], axis=1)
    # reorder columns to match observed
    df = df.loc[:, [MONTHS_SINCE_LSTG, BYR_HIST]]
    return df


def concat_sim_chunks(part, lookup, drop_censored=True):
    """
    Loops over simulations, concatenates dataframes.
    :param part: string name of partition.
    :param lookup: dataframe of listing values for part.
    :param drop_censored: True if censored observations are dropped.
    :return: dictionary of dataframes.
    """
    # collect chunks
    threads, offers = [], []
    for i in range(1, SIM_CHUNKS + 1):
        sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
        threads.append(sim['threads'])
        offers.append(sim['offers'])

    # concatenate
    threads = pd.concat(threads, axis=0).sort_index()
    offers = pd.concat(offers, axis=0).sort_index()

    # drop censored offers
    if drop_censored:
        offers = offers.loc[~offers[CENSORED], :]

    # initialize output dictionary
    sim = dict()

    # end of listing
    sale_time = offers.loc[offers[CON] == 100, 'clock'].reset_index(
        level=['thread', 'index'], drop=True)
    lstg_end = sale_time.reindex(index=lookup.index, fill_value=-1)
    no_sale = lstg_end[lstg_end == -1].index
    lstg_end.loc[no_sale] = lookup.loc[no_sale, 'start_time'] + MONTH - 1

    # conform to observed inputs
    sim['threads'] = process_sim_threads(threads, lookup.start_time)
    sim['offers'], clock = process_sim_offers(offers, lstg_end)

    # timestamps
    sim['thread_start'] = clock.xs(1, level='index')
    sim['lstg_end'] = lstg_end

    return sim


def get_obs_outcomes(part, lookup, drop_censored=True):
    # initialize output dictionary
    obs = dict()

    # observed outcomes
    obs['threads'] = load_file(part, 'x_thread')
    obs['offers'] = load_file(part, 'x_offer')

    if drop_censored:
        keep = (obs['offers'][DELAY] == 1) | ~obs['offers'][EXP]
        obs['offers'] = obs['offers'][keep]

    # timestamps
    obs['thread_start'] = load_file(part, 'clock').xs(1, level='index')
    obs['lstg_end'] = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(
                           index=lookup.index)

    return obs


def save_discrim_files(part, name, x_obs, x_sim):
    # featnames and sizes
    if part == VALIDATION:
        save_sizes(x_obs, name)

    # indices
    idx_obs = x_obs['lstg'].index
    idx_sim = x_sim['lstg'].index

    # create dictionary of numpy arrays
    x_obs = convert_x_to_numpy(x_obs, idx_obs)
    x_sim = convert_x_to_numpy(x_sim, idx_sim)

    # y=1 for real data
    y_obs = np.ones(len(idx_obs), dtype=bool)
    y_sim = np.zeros(len(idx_sim), dtype=bool)
    d = {'y': np.concatenate((y_obs, y_sim), axis=0)}

    # join input variables
    assert x_obs.keys() == x_sim.keys()
    d['x'] = {k: np.concatenate((x_obs[k], x_sim[k]), axis=0) for k in x_obs.keys()}

    # save inputs
    dump(d, INPUT_DIR + '{}/{}.gz'.format(part, name))

    # save small
    if part == TRAIN_RL:
        save_small(d, name)
