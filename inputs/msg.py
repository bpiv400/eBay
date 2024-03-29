import argparse
import pandas as pd
from inputs.util import save_files, check_zero, get_x_thread, get_ind_x
from utils import load_file
from constants import IDX
from featnames import CON, MSG, AUTO, EXP, TIME_FEATS, LOOKUP, BYR, \
    X_THREAD, X_OFFER, THREAD, INDEX, SIM_PARTITIONS


def get_y(offers, turn):
    # subset to turn
    df = offers.xs(turn, level=INDEX)
    # for buyers, drop accepts and rejects
    if turn in IDX[BYR]:
        mask = (df[CON] > 0) & (df[CON] < 1)
    # for sellers, drop accepts, expires, and auto responses
    else:
        mask = ~df[EXP] & ~df[AUTO] & (df[CON] < 1)
    return df.loc[mask, MSG]


def get_x_offer(offers=None, idx=None, turn=None):
    # initialize dictionary of offer features
    x_offer = {}
    # dataframe of offer features for relevant threads
    offers = pd.DataFrame(index=idx).join(offers)
    # drop time feats from buyer models
    if turn in IDX[BYR]:
        offers.drop(TIME_FEATS, axis=1, inplace=True)
    # turn features
    for i in range(1, turn + 1):
        # offer features at turn i
        offer = offers.xs(i, level=INDEX).reindex(
            index=idx, fill_value=0).astype('float32')
        # set unseen feats to 0
        if i == turn:
            offer[MSG] = 0
        # put in dictionary
        x_offer['offer{}'.format(i)] = offer.astype('float32')

    # error checking
    check_zero(x_offer)

    return x_offer


def process_inputs(part, turn):
    threads = load_file(part, X_THREAD)
    offers = load_file(part, X_OFFER)
    lstgs = load_file(part, LOOKUP).index

    # y and master index
    y = get_y(offers, turn)
    idx = y.index

    # thread features
    x = {THREAD: get_x_thread(threads, idx)}

    # offer features
    x.update(get_x_offer(offers, idx, turn))

    # indices for listing features
    idx_x = get_ind_x(lstgs=lstgs, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=SIM_PARTITIONS)
    parser.add_argument('--turn', type=int, choices=range(1, 7))
    args = parser.parse_args()
    part, turn = args.part, args.turn

    # model name
    name = '{}{}'.format(MSG, turn)
    print('{}/{}'.format(part, name))

    # input dataframes, output processed dataframes
    d = process_inputs(part, turn)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()
