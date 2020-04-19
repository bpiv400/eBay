import argparse
import pandas as pd
from processing.processing_utils import get_obs_outcomes
from processing.e_inputs.offer import process_inputs
from processing.f_discrim.discrim_utils import concat_sim_chunks, \
    save_discrim_files
from processing.f_discrim.discrim_consts import OFFER_FEATS
from constants import OFFER_MODELS, TRAIN_RL, VALIDATION, TEST


def construct_x_offer(d, offers, feats, turn):
    # x and y
    x = d['x']
    y = offers[feats].reindex(index=d['y'].index)
    # feature group, initialize if necessary
    group = 'offer{}'.format(turn)
    if group not in x:
        x[group] = pd.DataFrame(0.0,
                                index=x['offer1'].index,
                                columns=x['offer1'].columns)
    # add features to group
    for c in y.columns:
        if c in x[group]:
            x[group][c] = y[c].astype('float32')
    # check for nans
    assert x[group].isna().sum().sum() == 0
    return x


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    part, m = args.part, args.name
    name = '{}_discrim'.format(m)
    print('{}/{}'.format(part, name))

    # error check parameters
    assert part in [TRAIN_RL, VALIDATION, TEST]
    assert m in OFFER_MODELS

    # outcome and turn
    outcome = m[:-1]
    turn = int(m[-1])

    # dictionaries of components
    obs = get_obs_outcomes(part)
    sim = concat_sim_chunks(part)

    # offers for turn
    offers_obs = obs['offers'].xs(turn, level='index')
    offers_sim = sim['offers'].xs(turn, level='index')

    # dictionaries with x and y
    d_obs = process_inputs(obs, part, outcome, turn)
    d_sim = process_inputs(sim, part, outcome, turn)

    # put y in x
    feats = OFFER_FEATS[outcome]
    x_obs = construct_x_offer(d_obs, offers_obs, feats, turn)
    x_sim = construct_x_offer(d_sim, offers_sim, feats, turn)

    # save various output files
    save_discrim_files(part, name, x_obs, x_sim)


if __name__ == '__main__':
    main()
