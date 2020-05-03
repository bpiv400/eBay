import argparse
import pandas as pd
from processing.processing_utils import init_x, get_x_thread, load_file
from processing.f_discrim.discrim_utils import save_discrim_files
from constants import TRAIN_RL, VALIDATION, TEST
from featnames import SPLIT, DAYS, DELAY, EXP, AUTO, REJECT, \
    TIME_FEATS, MSG


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


def construct_x(part, threads, offers):
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


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--tf', action='store_true')
    args = parser.parse_args()
    part, tf = args.part, args.tf
    assert part in [TRAIN_RL, VALIDATION, TEST]
    name = 'threads' if tf else 'threads_no_tf'
    print('{}/{}'.format(part, name))

    # observed data
    threads_obs = load_file(part, 'x_thread')
    offers_obs = load_file(part, 'x_offer')

    # simulated data, first listing window only
    threads_sim = load_file(part, 'x_thread_sim').xs(0, level='sim')
    offers_sim = load_file(part, 'x_offer_sim').xs(0, level='sim')

    # construct input variable dictionaries
    x_obs = construct_x(part, threads_obs, offers_obs)
    x_sim = construct_x(part, threads_sim, offers_sim)

    # remove time feats
    if not tf:
        for i in range(1, 8):
            key = 'offer{}'.format(i)
            x_obs[key].drop(TIME_FEATS, axis=1, inplace=True)
            x_sim[key].drop(TIME_FEATS, axis=1, inplace=True)

    # save data
    save_discrim_files(part, name, x_obs, x_sim)


if __name__ == '__main__':
    main()
