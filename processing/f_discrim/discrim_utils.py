import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from compress_pickle import load, dump
from processing.e_inputs.inputs_utils import save_sizes, \
    convert_x_to_numpy, save_small
from processing.processing_utils import load_file, \
    collect_date_clock_feats, get_days_delay, get_norm
from utils import is_split, load_model
from train.train_consts import MBSIZE
from constants import TRAIN_RL, VALIDATION, INPUT_DIR, SIM_CHUNKS, \
    ENV_SIM_DIR, IDX, SLR_PREFIX, MONTH, NO_ARRIVAL_CUTOFF
from featnames import DAYS, DELAY, CON, SPLIT, NORM, REJECT, AUTO, EXP, \
    CENSORED, CLOCK_FEATS, TIME_FEATS, OUTCOME_FEATS, MONTHS_SINCE_LSTG, \
    BYR_HIST


def process_sim_offers(df, end_time):
    # censor timestamps
    timestamps = df.clock.to_frame().join(end_time.rename('end'))
    timestamps = timestamps.reorder_levels(df.index.names)
    clock = timestamps.min(axis=1)
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
    # select and sort features
    df = df.loc[:, CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS]
    return df, clock


def process_sim_threads(df, start_time):
    # convert clock to months_since_lstg
    df = df.join(start_time)
    df[MONTHS_SINCE_LSTG] = (df.clock - df.start_time) / MONTH
    df = df.drop(['clock', 'start_time'], axis=1)
    # reorder columns to match observed
    df = df.loc[:, [MONTHS_SINCE_LSTG, BYR_HIST]]
    return df


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
    # censored feats to 0
    df.loc[df[CENSORED], TIME_FEATS] = 0
    return df


def concat_sim_chunks(part, drop_censored=False,
                      restrict_to_first=True,
                      drop_no_arrivals=False):
    """
    Loops over simulations, concatenates dataframes.
    :param str part: name of partition.
    :param bool drop_censored: True if censored observations are dropped.
    :param bool restrict_to_first: True if only first simulation is kept.
    :param bool drop_no_arrivals: True if dropping listings with infrequent arrivals.
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

    # load lookup
    lookup = load_file(part, 'lookup')

    # initialize output dictionary
    sim = dict()

    # drop listings with infrequent arrivals
    if drop_no_arrivals:
        lookup = lookup.loc[lookup.p_no_arrival < NO_ARRIVAL_CUTOFF, :]
        threads = threads.reindex(index=lookup.index, level='lstg')
        offers = offers.reindex(index=lookup.index, level='lstg')
        sim['lookup'] = lookup

    # keep only first simulation
    if restrict_to_first:
        threads = threads.xs(0, level='sim')
        offers = offers.xs(0, level='sim')

    # difference time feats
    offers = diff_tf(offers)

    # drop censored offers
    if drop_censored:
        offers = offers.loc[~offers[CENSORED], :]

    # index of 'lstg' or ['lstg', 'sim']
    idx = threads.reset_index('thread', drop=True).index.unique()

    # end of listing
    sale_time = offers.loc[offers[CON] == 100, 'clock'].reset_index(
        level=['thread', 'index'], drop=True)
    sim['end_time'] = sale_time.reindex(index=idx, fill_value=-1)
    no_sale = sim['end_time'][sim['end_time'] == -1].index
    sim['end_time'].loc[no_sale] = sim['end_time'].loc[no_sale] + MONTH - 1

    # conform to observed inputs
    sim['threads'] = process_sim_threads(threads, lookup.start_time)
    sim['offers'], sim['clock'] = process_sim_offers(offers, sim['end_time'])

    return sim


def save_discrim_files(part, name, x_obs, x_sim):
    """
    Packages discriminator inputs for training.
    :param part: string name of partition.
    :param name: string name of model.
    :param x_obs: dictionary of observed data.
    :param x_sim: dictionary of simulated data.
    :return: None
    """
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


def get_model_predictions(m, x):
    # initialize neural net
    net = load_model(m, verbose=False)
    if torch.cuda.is_available():
        net = net.to('cuda')

    # split into batches
    v = np.array(range(len(x['lstg'])))
    batches = np.array_split(v, 1 + len(v) // MBSIZE[False])

    # model predictions
    p0 = []
    for b in batches:
        x_b = {k: torch.from_numpy(v[b, :]) for k, v in x.items()}
        if torch.cuda.is_available():
            x_b = {k: v.to('cuda') for k, v in x_b.items()}
        theta_b = net(x_b).cpu().double()
        p0.append(np.exp(log_softmax(theta_b, dim=-1)))

    # concatenate and return
    return torch.cat(p0, dim=0).numpy()