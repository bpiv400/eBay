import argparse
import pandas as pd
from processing.processing_utils import init_x, get_x_thread, \
    get_obs_outcomes
from processing.f_discrim.discrim_utils import concat_sim_chunks, \
    save_discrim_files
from processing.processing_consts import INTERVAL, INTERVAL_COUNTS
from constants import TRAIN_RL, VALIDATION, TEST, DAY, MAX_DELAY
from featnames import SPLIT, DAYS, DELAY, EXP, AUTO, REJECT, \
    TIME_FEATS, MSG, TIME_OF_DAY


def get_x_offer(offers, idx, tf):
    # initialize dictionary of offer features
    x_offer = dict()
    # turn features
    for i in range(1, 8):
        # offer features at turn i
        offer = offers.xs(i, level='index').reindex(
            index=idx, fill_value=0)
        # drop time feats, if tf parameter is False
        if not tf:
            offer.drop(TIME_FEATS, axis=1, inplace=True)
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
        # drop time of day
        offer.drop(TIME_OF_DAY, axis=1, inplace=True)
        # days and delay in intervals
        if i > 1:
            seconds = offer[DAYS] * DAY
            intervals = seconds // INTERVAL[i]
            offer.loc[:, DELAY] = intervals / INTERVAL_COUNTS[i]
            offer.loc[:, DAYS] = offer[DELAY] * MAX_DELAY[i] / DAY
        # put in dictionary
        x_offer['offer%d' % i] = offer.astype('float32')
    return x_offer


def construct_x(part, tf, d):
    # master index
    idx = d['threads'].index
    # initialize input dictionary with lstg features
    x = init_x(part, idx)
    # add thread features to x['lstg']
    x_thread = get_x_thread(d['threads'], idx, censor_months=True)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)
    # offer features
    x.update(get_x_offer(d['offers'], idx, tf))
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
    obs = get_obs_outcomes(part, drop_censored=True)
    x_obs = construct_x(part, tf, obs)

    # simulated data
    sim = concat_sim_chunks(part, drop_censored=True)
    x_sim = construct_x(part, tf, sim)

    # save data
    save_discrim_files(part, name, x_obs, x_sim)


if __name__ == '__main__':
    main()
