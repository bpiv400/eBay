import argparse
import os
import pandas as pd
from agent.util import get_log_dir, load_values, \
    get_byr_valid, get_byr_return, get_slr_valid, get_slr_return
from utils import topickle, load_data, get_role, compose_args
from agent.const import AGENT_PARAMS
from constants import HEURISTIC_DIR, AGENT_PARTITIONS, TEST
from featnames import SIM, OBS, X_OFFER


def get_return(data=None, values=None, byr=False):
    # restrict to valid and calculate rewards
    if byr:
        data = get_byr_valid(data)
        reward = get_byr_return(data=data, values=values)
    else:
        data = get_slr_valid(data)
        reward = get_slr_return(data=data, values=values)

    return reward


def wrapper(values=None, byr=False):
    return lambda d: get_return(data=d, values=values, byr=byr)


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    args = parser.parse_args()

    # preliminaries
    values = load_values(part=args.part, delta=args.delta)
    if not args.byr:
        values *= args.delta
    f = wrapper(values=values, byr=args.byr)
    df = pd.DataFrame(columns=['norm', 'dollar'])  # for output

    # rewards from data
    for dset in [OBS]:
        data = load_data(part=args.part, sim=(dset == SIM))
        df.loc[dset, :] = f(data)

    # rewards from heuristic strategy
    heuristic_dir = HEURISTIC_DIR + '{}/'.format(get_role(args.byr))
    data = load_data(part=args.part, run_dir=heuristic_dir)
    df.loc['heuristic', :] = f(data)

    # rewards from agent runs
    log_dir = get_log_dir(byr=args.byr, delta=args.delta)
    run_ids = [p for p in os.listdir(log_dir) if os.path.isdir(log_dir + p)]
    for run_id in run_ids:
        run_dir = log_dir + '{}/'.format(run_id)
        data = load_data(part=args.part, run_dir=run_dir)
        if X_OFFER in data:
            df.loc[run_id, :] = f(data)

    # save table
    print(df)
    topickle(df, log_dir + '{}.pkl'.format(args.part))


if __name__ == '__main__':
    main()
