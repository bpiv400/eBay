import numpy as np
from agent.util import get_sim_dir, load_valid_data
from constants import INPUT_DIR
from inputs.util import get_x_thread, save_featnames_and_sizes, convert_x_to_numpy, \
    get_ind_x
from utils import load_file, input_partition, load_data, topickle
from featnames import LOOKUP, THREAD, X_THREAD, X_OFFER, DISCRIM_MODEL, \
    PLACEBO_MODEL, SIM, INDEX, DAYS, DELAY, EXP, REJECT, AUTO, MSG, COMMON, VALIDATION


def save_discrim_files(part=None, model=None, x_obs=None, x_sim=None, lstgs=None):
    """
    Packages discriminator inputs for training.
    :param str part: string name of partition
    :param str model: either DISCRIM_MODEL or PLACEBO_MODEL
    :param dict x_obs: dictionary of observed data
    :param dict x_sim: dictionary of simulated data
    :param pd.Index lstgs: index of lstg ids
    :return: None
    """
    # featnames and sizes
    if part == VALIDATION:
        save_featnames_and_sizes(x_obs, model)

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
    topickle(d, INPUT_DIR + '{}/{}.pkl'.format(part, model))


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
            for feat in [MSG, COMMON]:
                assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
                offer.drop(feat, axis=1, inplace=True)
        # put in dictionary
        x_offer['offer{}'.format(i)] = offer
    return x_offer


def construct_x(data=None):
    idx = data[X_THREAD].index  # master index
    x = {THREAD: get_x_thread(data[X_THREAD], idx)}  # thread features
    x.update(get_x_offer(data[X_OFFER], idx))  # offer features
    return x


def import_data(part=None, sim=False, placebo=False):
    # load files
    if placebo:
        if sim:
            sim_dir = get_sim_dir(byr=True, delta=1)
            data = load_valid_data(part=part, sim_dir=sim_dir, lookup=False)
        else:
            data = load_valid_data(part=part, byr=True, lookup=False)
    else:
        data = load_data(part=part, sim=sim, lookup=False)
    # only use first sim
    if sim:
        for k, v in data.items():
            data[k] = v.xs(0, level=SIM)
    return data


def main():
    # extract parameters from command line
    part, placebo = input_partition(agent=True, opt_arg='placebo')
    model = PLACEBO_MODEL if placebo else DISCRIM_MODEL
    print('{}/{}'.format(part, model))

    # listing ids
    lstgs = load_file(part, LOOKUP).index

    # data files
    data_obs = import_data(part=part, sim=False, placebo=placebo)
    data_sim = import_data(part=part, sim=True, placebo=placebo)

    # construct input variable dictionaries
    x_obs = construct_x(data=data_obs)
    x_sim = construct_x(data=data_sim)

    # save data
    save_discrim_files(part=part,
                       x_obs=x_obs,
                       x_sim=x_sim,
                       lstgs=lstgs,
                       model=model)


if __name__ == '__main__':
    main()
