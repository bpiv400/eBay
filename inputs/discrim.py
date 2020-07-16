import argparse
import numpy as np
import pandas as pd
from compress_pickle import dump
from inputs.util import save_featnames_and_sizes, \
    convert_x_to_numpy, get_x_thread, get_ind_x
from utils import load_file, drop_censored
from constants import VALIDATION, DISCRIM_MODELS, AGENT_PARTITIONS, \
    DISCRIM_LISTINGS, DISCRIM_THREADS_NO_TF, INPUT_DIR, MONTH
from featnames import SPLIT, DAYS, DELAY, EXP, AUTO, REJECT, \
    TIME_FEATS, MSG, START_TIME, CON, MONTHS_SINCE_LSTG, END_TIME, LOOKUP

AGENT = True


def save_discrim_files(part=None, name=None, x_obs=None, x_sim=None, lstgs=None):
    """
    Packages discriminator inputs for training.
    :param part: string name of partition.
    :param name: string name of model.
    :param x_obs: dictionary of observed data.
    :param x_sim: dictionary of simulated data.
    :param lstgs: index of lstg ids.
    :return: None
    """
    # featnames and sizes
    if part == VALIDATION:
        save_featnames_and_sizes(x_obs, name)

    # indices
    idx_obs = x_obs['thread'].index
    idx_sim = x_sim['thread'].index

    # create dictionary of numpy arrays
    convert_x_to_numpy(x_obs, idx_obs)
    convert_x_to_numpy(x_sim, idx_sim)

    # y=1 for real data
    y_obs = np.ones(len(idx_obs), dtype=bool)
    y_sim = np.zeros(len(idx_sim), dtype=bool)
    d = {'y': np.concatenate((y_obs, y_sim), axis=0)}

    # join input variables
    assert x_obs.keys() == x_sim.keys()
    d['x'] = {k: np.concatenate((x_obs[k], x_sim[k]), axis=0)
              for k in x_obs.keys()}

    # indices for fixed features
    idx_x_obs = get_ind_x(lstgs=lstgs, idx=idx_obs)
    idx_x_sim = get_ind_x(lstgs=lstgs, idx=idx_sim)
    d['idx_x'] = np.concatenate((idx_x_obs, idx_x_sim), axis=0)

    # save inputs
    dump(d, INPUT_DIR + '{}/{}.gz'.format(part, name))


def get_x_offer(offers, idx):
    # initialize dictionary of offer features
    x_offer = dict()
    # turn features
    for i in range(1, 8):
        # offer features at turn i
        offer = offers.xs(i, level='index').reindex(
            index=idx, fill_value=0)
        # drop feats that are zero
        if i == 1:
            for feat in [DAYS, DELAY, EXP, REJECT]:
                assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
                offer.drop(feat, axis=1, inplace=True)
        if i % 2 == 1:
            for feat in [AUTO]:
                assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
                offer.drop(feat, axis=1, inplace=True)
        if i == 7:
            for feat in [MSG, SPLIT]:
                assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
                offer.drop(feat, axis=1, inplace=True)
        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')
    return x_offer


def construct_x_threads(threads, offers):
    # master index
    idx = threads.index
    # thread features
    x = {'thread': get_x_thread(threads, idx)}
    # offer features
    x.update(get_x_offer(offers, idx))
    return x


def construct_x_listings(idx_lstg=None, idx_thread=None):
    x_thread = pd.Series(True, index=idx_thread, name='has_thread')
    x_thread = x_thread.reindex(index=idx_lstg, fill_value=False)
    return {'thread': x_thread.to_frame()}


def clean_sim_offers(offers, months, part):
    # drop offers after expiration
    clock_sim = load_file(part, 'clock_sim', agent=AGENT).reindex(
        index=offers.index)
    start_time = load_file(part, 'lookup', agent=AGENT)[START_TIME]
    months_sim = (clock_sim - start_time.reindex(
        index=clock_sim.index, level='lstg')) / MONTH
    keep = months_sim < months.reindex(
        index=months_sim.index, level='lstg', fill_value=1.)
    return offers[keep]


def clean_sim_threads(threads, months):
    keep = threads[MONTHS_SINCE_LSTG] < months.reindex(
        index=threads.index, level='lstg', fill_value=1.)
    return threads[keep]


def months_to_exp(part):
    is_sale = (load_file(part, 'x_offer', agent=AGENT)[CON] == 1).groupby(
        'lstg').max()
    exp_lstgs = is_sale[~is_sale].index
    lookup = load_file(part, LOOKUP, agent=AGENT)
    exp_time = lookup.loc[exp_lstgs, END_TIME]
    start_time = lookup.loc[exp_lstgs, START_TIME]
    months = (exp_time - start_time + 1) / MONTH
    months = months[months < 1.]
    return months


def load_threads_offers(part=None, sim=False):
    suffix = '_sim' if sim else ''
    threads = load_file(part, 'x_thread{}'.format(suffix), agent=AGENT)
    offers = load_file(part, 'x_offer{}'.format(suffix), agent=AGENT)
    offers = drop_censored(offers)
    if sim:  # drop threads and offers after observed lstg expiration
        months = months_to_exp(part)
        threads = clean_sim_threads(threads, months)
        offers = clean_sim_offers(offers, months, part)
    return threads, offers


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--name', choices=DISCRIM_MODELS, required=True)
    args = parser.parse_args()
    part, name = args.part, args.name
    assert part in AGENT_PARTITIONS
    print('{}/{}'.format(part, name))

    # listing ids
    lstgs = load_file(part, LOOKUP, agent=AGENT).index

    # observed and simulated outcomes
    threads_obs, offers_obs = load_threads_offers(part=part, sim=False)
    threads_sim, offers_sim = load_threads_offers(part=part, sim=True)

    # listings inputs
    if name == DISCRIM_LISTINGS:
        # observed data
        idx_thread_obs = threads_obs.xs(1, level='thread').index
        x_obs = construct_x_listings(idx_lstg=lstgs,
                                     idx_thread=idx_thread_obs)

        # simulated data
        idx_thread_sim = threads_sim.xs(1, level='thread').index
        x_sim = construct_x_listings(idx_lstg=lstgs,
                                     idx_thread=idx_thread_sim)

    # threads inputs
    else:
        # construct input variable dictionaries
        x_obs = construct_x_threads(threads_obs, offers_obs)
        x_sim = construct_x_threads(threads_sim, offers_sim)

        # remove time feats
        if name == DISCRIM_THREADS_NO_TF:
            for i in range(1, 8):
                key = 'offer{}'.format(i)
                x_obs[key].drop(TIME_FEATS, axis=1, inplace=True)
                x_sim[key].drop(TIME_FEATS, axis=1, inplace=True)

    # save data
    save_discrim_files(part=part,
                       name=name,
                       x_obs=x_obs,
                       x_sim=x_sim,
                       lstgs=lstgs)


if __name__ == '__main__':
    main()
