import argparse
import os
import pandas as pd
from agent.util import get_log_dir
from assess.util import get_valid_slr, get_value_slr
from utils import unpickle, load_file, topickle
from constants import VALIDATION, RL_SLR, PARTS_DIR
from featnames import LOOKUP, START_PRICE, OBS, X_OFFER

PARTS = [VALIDATION, RL_SLR]


def get_values(folder=None, lookup=None):
    path = folder + '{}.pkl'.format(X_OFFER)
    if os.path.isfile(path):
        data = dict(offers=unpickle(path))
        data, lookup = get_valid_slr(data=data, lookup=lookup)
        return get_value_slr(offers=data['offers'],
                             start_price=lookup[START_PRICE])
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

    # initialize output dataframe
    run_ids = [p for p in os.listdir(log_dir) if os.path.isdir(log_dir + p)]
    idx = pd.MultiIndex.from_product([PARTS, [OBS] + run_ids],
                                     names=['part', 'run_id'])
    df = pd.DataFrame(columns=['norm', 'price'], index=idx)

    for part in PARTS:
        lookup = load_file(part, LOOKUP)

        # values from data
        folder = PARTS_DIR + '{}/'.format(part)
        df.loc[(part, OBS), :] = get_values(folder=folder, lookup=lookup)

        # values from agents runs
        for run_id in run_ids:
            folder = log_dir + '{}/{}/'.format(run_id, part)
            if args.heuristic:
                folder += 'heuristic/'
            values = get_values(folder=folder, lookup=lookup)
            if values is not None:
                df.loc[(part, run_id), :] = values

    df = df.sort_index()

    # save table
    print(df)
    topickle(df, log_dir + 'runs.pkl')


if __name__ == '__main__':
    main()
