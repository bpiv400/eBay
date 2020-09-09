import argparse
import os
import pandas as pd
from agent.util import get_log_dir, get_valid_slr, get_value_slr
from utils import unpickle, load_file, topickle
from constants import VALIDATION, RL_SLR, PARTS_DIR, DAY
from featnames import LOOKUP, START_PRICE, SIM, OBS, X_OFFER, END_TIME

PARTS = [VALIDATION, RL_SLR]
FIXED_RUNS = [OBS, SIM, 'noslrexp', 'slraccrej']


def get_values(folder=None, part=None, subfolder='', restricted=False):
    lookup = load_file(part, LOOKUP)

    # restrict to sales
    if restricted:
        no_sale = (lookup[END_TIME] + 1) % DAY == 0
        lookup = lookup[~no_sale]

    path = folder + '{}/{}{}.pkl'.format(part, subfolder, X_OFFER)
    if os.path.isfile(path):
        data = dict(offers=unpickle(path))
        data, lookup = get_valid_slr(data=data, lookup=lookup)
        return get_value_slr(offers=data['offers'],
                             start_price=lookup[START_PRICE])
    else:
        return None


def populate_df(df=None, part=None, log_dir=None, run_ids=None,
                restricted=False, heuristic=False):
    lstg_set = 'sales' if restricted else 'all'

    # values from data
    for dset in FIXED_RUNS:
        df.loc[(part, lstg_set, dset), :] = get_values(
            folder=PARTS_DIR,
            part=part,
            subfolder='{}/'.format(dset) if dset != OBS else '',
            restricted=restricted
        )

    # values from agents runs
    for run_id in run_ids:
        values = get_values(folder=log_dir + '{}/'.format(run_id),
                            part=part,
                            subfolder='heuristic/' if heuristic else '',
                            restricted=restricted)
        if values is not None:
            df.loc[(part, lstg_set, run_id), :] = values


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
    idx = pd.MultiIndex.from_product([PARTS, ['all', 'sales'], FIXED_RUNS + run_ids],
                                     names=['part', 'listings', 'run_id'])
    df = pd.DataFrame(columns=['norm', 'price'], index=idx).sort_index()

    for part in PARTS:
        for restricted in [False, True]:
            populate_df(df=df,
                        part=part,
                        log_dir=log_dir,
                        run_ids=run_ids,
                        restricted=restricted,
                        heuristic=args.heuristic)

    # save table
    print(df)
    topickle(df, log_dir + 'runs.pkl')


if __name__ == '__main__':
    main()
