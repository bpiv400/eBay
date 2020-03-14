import argparse
from compress_pickle import load
import pandas as pd
from processing.processing_utils import load_file, input_partition, \
    get_days_delay, get_norm, get_x_thread, collect_date_clock_feats
from processing.f_discrim.discrim_utils import concat_sim_chunks, save_discrim_files
from utils import is_split
from constants import MONTH, IDX, SLR_PREFIX, TRAIN_RL, VALIDATION, INPUT_DIR, INDEX_DIR
from featnames import CON, NORM, SPLIT, DAYS, DELAY, EXP, AUTO, REJECT, CENSORED, \
    MONTHS_SINCE_LSTG, TIME_FEATS, MSG, BYR_HIST


def initialize_inputs(part, turn):
    # toggle input name by turn
    input_name = (CON if turn == 1 else DELAY) + str(turn)
    # load components
    featnames = load(INPUT_DIR + '{}/{}.pkl'.format('featnames', input_name))
    idx = load(INDEX_DIR + '{}/{}.gz'.format(part, input_name))
    d = load(INPUT_DIR + '{}/{}.gz'.format(part, input_name))
    # construct x
    x = dict()
    for k, v in d['x'].items():
        fkey = 'offer' if 'offer' in k else k
        x[k] = pd.DataFrame(v, index=idx, columns=featnames[fkey])
    # for turn 1, delete offer1
    if turn == 1:
        del x['offer1']
    return x


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--turn', type=int)
    args = parser.parse_args()
    part, turn = args.part, args.turn

    # error check inputs
    assert part in [TRAIN_RL, VALIDATION]
    assert turn in range(1, 8)
    print('{}/turn{}'.format(part, turn))

    # initialize discriminator inputs
    x = initialize_inputs(part, turn)

    # current offer, observed and simulated
    offer_obs = get_current_offer()