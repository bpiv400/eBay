import numpy as np
import pandas as pd
from compress_pickle import load, dump
from processing.processing_utils import save_sizes, convert_x_to_numpy, save_small
from constants import SIM_CHUNKS, ENV_SIM_DIR, MAX_DELAY, ARRIVAL_PREFIX, INPUT_DIR


def concat_sim_chunks(part):
    """
    Loops over simulations, concatenates dataframes.
    :param part: string name of partition.
    :return: concatentated and sorted threads and offers dataframes.
    """
    threads, offers = [], []
    for i in range(1, SIM_CHUNKS + 1):
        sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
        threads.append(sim['threads'])
        offers.append(sim['offers'])
    threads = pd.concat(threads, axis=0).sort_index()
    offers = pd.concat(offers, axis=0).sort_index()
    return threads, offers


def process_lstg_end(lstg_start, lstg_end):
    # remove thread and index from lstg_end index
    lstg_end = lstg_end.reset_index(['thread', 'index'], drop=True)
    assert not lstg_end.index.duplicated().max()

    # fill in missing lstg end times with expirations
    lstg_end = lstg_end.reindex(index=lstg_start.index, fill_value=-1)
    lstg_end.loc[lstg_end == -1] = lstg_start + MAX_DELAY[ARRIVAL_PREFIX] - 1

    return lstg_end


def get_sim_times(part, lstg_start):
    # collect times from simulation files
    lstg_end, thread_start = [], []
    for i in range(1, SIM_CHUNKS + 1):
        sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
        offers, threads = [sim[k] for k in ['offers', 'threads']]
        lstg_end.append(offers.loc[(offers.con == 100) & ~offers.censored, 'clock'])
        thread_start.append(threads.clock)

    # concatenate into single series
    lstg_end = pd.concat(lstg_end, axis=0).sort_index()
    thread_start = pd.concat(thread_start, axis=0).sort_index()

    # shorten index and fill-in expirations
    lstg_end = process_lstg_end(lstg_start, lstg_end)

    return lstg_end, thread_start


def save_discrim_files(part, name, x_obs, x_sim):
    # featnames and sizes
    if part == 'test_rl':
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
    if part == 'train_rl':
        save_small(d, name)
