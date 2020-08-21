import argparse
import os
from compress_pickle import load
import numpy as np
import pandas as pd
from agent.util import get_log_dir
from assess.util import get_value_slr
from utils import load_file, topickle
from constants import VALIDATION
from featnames import LOOKUP, START_PRICE, OBS


def print_summary(df, idx):
    v_norm = np.round(100 * df.loc[idx, 'norm'], decimals=1)
    v_price = np.round(df.loc[idx, 'price'], decimals=2)
    print('{}: {}%, ${}'.format(idx, v_norm, v_price))


def main():
    # buyer flag
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--heuristic', action='store_true')
    args = parser.parse_args()
    log_dir = get_log_dir(byr=args.byr)

    if args.byr:
        raise NotImplementedError()

    start_price = load_file(VALIDATION, LOOKUP)[START_PRICE]

    # initialize with values from data
    df = pd.DataFrame(columns=['norm', 'price'], dtype=np.float64)
    offers = load_file(VALIDATION, 'x_offer')
    df.loc[OBS, :] = get_value_slr(offers=offers, start_price=start_price)
    print_summary(df, OBS)

    # agent runs
    for run_id in os.listdir(log_dir):
        run_dir = log_dir + '{}/'.format(run_id)
        if not os.path.isdir(run_dir):
            continue
        folder = run_dir + '{}/'.format(VALIDATION)
        if args.heuristic:
            folder += 'heuristic/'
        path = folder + 'x_offer.gz'
        if os.path.isfile(path):
            offers = load(path)
            df.loc[run_id] = get_value_slr(offers=offers,
                                           start_price=start_price)
            print_summary(df, run_id)
        else:
            print('{}: simulation output not found.'.format(run_id))

    # save table
    topickle(df, log_dir + 'runs.pkl')


if __name__ == '__main__':
    main()
