import argparse
import numpy as np
import pandas as pd
from compress_pickle import dump
from inputs.util import save_featnames, save_sizes, \
    convert_x_to_numpy, save_small, get_x_thread
from utils import load_file, init_x, drop_censored
from constants import TRAIN_RL, VALIDATION, TEST, DISCRIM_MODELS, \
    DISCRIM_LISTINGS, DISCRIM_THREADS_NO_TF, INPUT_DIR, MONTH
from featnames import SPLIT, DAYS, DELAY, EXP, AUTO, REJECT, \
    TIME_FEATS, MSG, START_TIME, CON, MONTHS_SINCE_LSTG


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
        save_featnames(x_obs, name)
        save_sizes(x_obs, name)

    # indices
    idx_obs = x_obs['lstg'].index
    idx_sim = x_sim['lstg'].index

    # create dictionary of numpy arrays
    convert_x_to_numpy(x_obs, idx_obs)
    convert_x_to_numpy(x_sim, idx_sim)

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


def construct_x_threads(part, threads, offers):
    # master index
    idx = threads.index
    # initialize input dictionary with lstg features
    x = init_x(part, idx)
    # add thread features to x['lstg']
    x_thread = get_x_thread(threads, idx)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)
    # offer features
    x.update(get_x_offer(offers, idx))
    return x


def construct_x_listings(x, idx_thread):
    d = x.copy()
    idx = x['lstg'].index
    has_thread = pd.Series(1.0, index=idx_thread, name='has_thread').reindex(
        index=idx, fill_value=0.0)
    d['lstg'] = d['lstg'].join(has_thread)
    return d


def months_to_exp(part):
    is_sale = (load_file(part, 'x_offer')[CON] == 1).groupby(
        'lstg').max()
    idx_sale = is_sale[is_sale].index
    exp_time = load_file(part, 'lstg_end').drop(idx_sale)
    start_time = load_file(part, 'lookup')[START_TIME].drop(idx_sale)
    months = (exp_time - start_time + 1) / MONTH
    months = months[months < 1.]
    return months


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--name', choices=DISCRIM_MODELS, required=True)
    args = parser.parse_args()
    part, name = args.part, args.name
    assert part in [TRAIN_RL, VALIDATION, TEST]
    print('{}/{}'.format(part, name))

    # drop listings that expire before a month in the data
    months = months_to_exp(part)

    # threads data, observed and simulated
    threads_obs = load_file(part, 'x_thread')
    threads_sim = load_file(part, 'x_thread_sim').xs(0, level='sim')

    # drop simulated threads after expiration
    keep = threads_sim[MONTHS_SINCE_LSTG] < months.reindex(
        index=threads_sim.index, level='lstg', fill_value=1.)
    threads_sim = threads_sim[keep]

    # listings inputs
    if name == DISCRIM_LISTINGS:
        # initialize listing features
        x = init_x(part)

        # observed data
        idx_obs = threads_obs.xs(1, level='thread').index
        x_obs = construct_x_listings(x, idx_obs)

        # simulated data
        idx_sim = threads_sim.xs(1, level='thread').index
        x_sim = construct_x_listings(x, idx_sim)

    # threads inputs
    else:
        # offers data, observed and simulated
        offers_obs = load_file(part, 'x_offer')
        offers_sim = load_file(part, 'x_offer_sim').xs(
            0, level='sim')

        # drop offers after expiration
        clock_sim = load_file(part, 'clock_sim').xs(0, level='sim')
        start_time = load_file(part, 'lookup')[START_TIME]
        months_sim = (clock_sim - start_time.reindex(
            index=clock_sim.index, level='lstg')) / MONTH
        keep = months_sim < months.reindex(
            index=months_sim.index, level='lstg', fill_value=1.)
        offers_sim = offers_sim[keep]

        # drop censored offers
        offers_obs = drop_censored(offers_obs)
        offers_sim = drop_censored(offers_sim)

        # construct input variable dictionaries
        x_obs = construct_x_threads(part, threads_obs, offers_obs)
        x_sim = construct_x_threads(part, threads_sim, offers_sim)

        # remove time feats
        if name == DISCRIM_THREADS_NO_TF:
            for i in range(1, 8):
                key = 'offer{}'.format(i)
                x_obs[key].drop(TIME_FEATS, axis=1, inplace=True)
                x_sim[key].drop(TIME_FEATS, axis=1, inplace=True)

    # save data
    save_discrim_files(part, name, x_obs, x_sim)


if __name__ == '__main__':
    main()
