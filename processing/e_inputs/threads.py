import argparse
import pandas as pd
from processing.processing_utils import load_file, concat_sim_chunks
from processing.e_inputs.inputs_utils import get_x_thread, init_x, save_discrim_files
from constants import TRAIN_RL, VALIDATION, TEST
from featnames import SPLIT, DAYS, DELAY, EXP, AUTO, REJECT, TIME_FEATS, MSG


def get_x_offer(offers, idx, tf):
    # initialize dictionary of offer features
    x_offer = {}
    # turn features
    for i in range(1, 8):
        # offer features at turn i
        offer = offers.xs(i, level='index').reindex(
            index=idx, fill_value=0).astype('float32')
        # drop time feats, if tf parameter is False
        if not tf:
            offer.drop(TIME_FEATS, axis=1, inplace=True)
        # drop feats that are zero
        if i == 1:
            for feat in [DAYS, DELAY, EXP, REJECT]:
                assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
                offer.drop(feat, axis=1, inplace=True)
        elif i % 2 == 1:
            for feat in [AUTO]:
                assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
                offer.drop(feat, axis=1, inplace=True)
        elif i == 7:
            for feat in [MSG, SPLIT]:
                assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
                offer.drop(feat, axis=1, inplace=True)
        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')
    return x_offer


def construct_x(part, tf, threads, offers):
    # master index
    idx = threads.index
    # initialize input dictionary with lstg features
    x = init_x(part, idx)
    # add thread features to x['lstg']
    x['lstg'] = pd.concat([x['lstg'], get_x_thread(threads, idx)], axis=1)
    # offer features
    x.update(get_x_offer(offers, idx, tf))
    return x


def main():
    # partiton
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--tf', action='store_true')
    args = parser.parse_args()
    part, tf = args.part, args.tf
    assert part in [TRAIN_RL, VALIDATION, TEST]
    name = 'threads' if tf else 'threads_no_tf'
    print('{}/{}}'.format(part, name))

    # observed data
    threads_obs = load_file(part, 'x_thread')
    offers_obs = load_file(part, 'x_offer')
    censored = (offers_obs[EXP] == 1) & (offers_obs[DELAY] < 1)
    offers_obs = offers_obs.loc[~censored, :]
    x_obs = construct_x(part, tf, threads_obs, offers_obs)

    # simulated data
    threads_sim, offers_sim = concat_sim_chunks(part)
    x_sim = construct_x(part, tf, threads_sim, offers_sim)

    # save data
    save_discrim_files(part, name, x_obs, x_sim)


if __name__ == '__main__':
    main()
