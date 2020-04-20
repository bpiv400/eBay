import argparse
import pandas as pd
from processing.processing_utils import get_obs_outcomes
from processing.e_inputs.offer import process_inputs
from processing.f_discrim.discrim_utils import concat_sim_chunks, \
    save_discrim_files
from processing.f_discrim.discrim_consts import OFFER_FEATS
from constants import OFFER_MODELS, TRAIN_RL, VALIDATION, TEST


def construct_x_offer(d, y, turn):
    # x and y
    x = d['x']
    y = y.reindex(index=d['y'].index)
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
    parser.add_argument('--part', type=str, required=True)
    parser.add_argument('--outcome', type=str, required=True)
    parser.add_argument('--turn', type=int, required=True)
    args = parser.parse_args()
    part, outcome, turn = args.part, args.outcome, args.turn
    m = outcome + str(turn)
    name = '{}_discrim'.format(m)
    print('{}/{}'.format(part, name))

    # error check parameters
    assert part in [TRAIN_RL, VALIDATION, TEST]
    assert m in OFFER_MODELS

    # dictionaries of components
    obs = get_obs_outcomes(part)
    sim = concat_sim_chunks(part)

    # dictionaries with x and y
    d_obs = process_inputs(obs, part, outcome, turn)
    d_sim = process_inputs(sim, part, outcome, turn)

    # offers for turn
    feats = OFFER_FEATS[outcome]
    y_obs = obs['offers'].xs(turn, level='index')[feats]
    y_sim = sim['offers'].xs(turn, level='index')[feats]

    # put y in x
    x_obs = construct_x_offer(d_obs, y_obs, turn)
    x_sim = construct_x_offer(d_sim, y_sim, turn)

    # save various output files
    save_discrim_files(part, name, x_obs, x_sim)


if __name__ == '__main__':
    main()
