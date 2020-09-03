import numpy as np
from inputs.util import save_featnames_and_sizes, \
    convert_x_to_numpy, get_x_thread, get_ind_x
from utils import topickle, load_file, input_partition
from constants import TRAIN_DISCRIM, VALIDATION, TEST, DISCRIM_MODEL, INPUT_DIR
from featnames import SPLIT, DAYS, DELAY, EXP, AUTO, REJECT, MSG, LOOKUP, THREAD, \
    X_OFFER, X_THREAD, INDEX


def save_discrim_files(part=None, x_obs=None, x_sim=None, lstgs=None):
    """
    Packages discriminator inputs for training.
    :param part: string name of partition.
    :param x_obs: dictionary of observed data.
    :param x_sim: dictionary of simulated data.
    :param lstgs: index of lstg ids.
    :return: None
    """
    # featnames and sizes
    if part == VALIDATION:
        save_featnames_and_sizes(x_obs, DISCRIM_MODEL)

    # indices
    idx_obs = x_obs[THREAD].index
    idx_sim = x_sim[THREAD].index

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
    topickle(d, INPUT_DIR + '{}/{}.pkl'.format(part, DISCRIM_MODEL))


def get_x_offer(offers, idx):
    # initialize dictionary of offer features
    x_offer = dict()
    # turn features
    for i in range(1, 8):
        # offer features at turn i
        offer = offers.xs(i, level=INDEX).reindex(
            index=idx, fill_value=0).astype('float32')
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
        x_offer['offer{}'.format(i)] = offer
    return x_offer


def load_threads_offers(part=None, sim=False):
    prefix = 'sim/' if sim else ''
    threads = load_file(part, '{}{}'.format(prefix, X_THREAD))
    offers = load_file(part, '{}{}'.format(prefix, X_OFFER))
    return threads, offers


def construct_x(part=None, sim=False):
    # load files
    threads, offers = load_threads_offers(part=part, sim=sim)
    # master index
    idx = threads.index
    # thread features
    x = {THREAD: get_x_thread(threads, idx)}
    # offer features
    x.update(get_x_offer(offers, idx))
    return x


def main():
    # extract parameters from command line
    part = input_partition()
    assert part in [TRAIN_DISCRIM, VALIDATION, TEST]
    print('{}/{}'.format(part, DISCRIM_MODEL))

    # listing ids
    lstgs = load_file(part, LOOKUP).index

    # construct input variable dictionaries
    x_obs = construct_x(part=part, sim=False)
    x_sim = construct_x(part=part, sim=True)

    # save data
    save_discrim_files(part=part,
                       x_obs=x_obs,
                       x_sim=x_sim,
                       lstgs=lstgs)


if __name__ == '__main__':
    main()
