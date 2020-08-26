import argparse
import os
import numpy as np
import pandas as pd
from agent.util import get_log_dir
from assess.util import get_value_slr
from utils import unpickle, load_file, topickle
from constants import VALIDATION, RL_SLR, PARTS_DIR
from featnames import LOOKUP, START_PRICE, OBS


def round_values(s=None, rl=False):
    cols = ['norm', 'price']
    if rl:
        cols = ['{}_rl'.format(c) for c in cols]
    v_norm = np.round(100 * s[cols[0]], decimals=1)
    v_price = np.round(s[cols[1]], decimals=2)
    return v_norm, v_price


def print_summary(df=None, idx=None):
    s = df.loc[idx]
    v_norm, v_price = round_values(s)
    print('{}: {}%, ${}'.format(idx, v_norm, v_price))

    if not np.isnan(s['norm_rl']):
        v_norm_rl, v_price_rl = round_values(s, rl=True)
        print('{} (rl): {}%, ${}'.format(idx, v_norm_rl, v_price_rl))


def get_values(run_dir=None, part=None, heuristic=False):
    folder = run_dir + '{}/'.format(part)
    if heuristic:
        folder += 'heuristic/'
    path = folder + 'x_offer.pkl'
    if os.path.isfile(path):
        offers = unpickle(path)
        start_price = load_file(part, LOOKUP)[START_PRICE]
        return get_value_slr(offers=offers, start_price=start_price)
    else:
        return None


def main():
    # buyer flag
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--heuristic', action='store_true')
    args = parser.parse_args()
    log_dir = get_log_dir(byr=args.byr)

    if args.byr:
        raise NotImplementedError()

    # initialize with values from data
    df = pd.DataFrame(columns=['norm', 'price', 'norm_rl', 'price_rl'],
                      dtype=np.float64)
    df.loc[OBS, ['norm', 'price']] = get_values(
        run_dir=PARTS_DIR, part=VALIDATION)
    print_summary(df, OBS)

    # agent runs
    for run_id in os.listdir(log_dir):
        run_dir = log_dir + '{}/'.format(run_id)
        values = get_values(run_dir=run_dir,
                            part=VALIDATION,
                            heuristic=args.heuristic)
        if values is None:
            print('{}: no simulation found'.format(run_id))
            continue
        df.loc[run_id, ['norm', 'price']] = values

        values_rl = get_values(run_dir=run_dir,
                               part=RL_SLR,
                               heuristic=args.heuristic)
        if values is not None:
            df.loc[run_id, ['norm_rl', 'price_rl']] = values_rl

        print_summary(df=df, idx=run_id)

    # save table
    topickle(df, log_dir + 'runs.pkl')


if __name__ == '__main__':
    main()
