import argparse
import os
import pandas as pd
from agent.util import get_log_dir, get_reward, load_values
from utils import topickle, load_data
from agent.const import DELTA
from constants import HOLDOUT_PARTITIONS
from featnames import SIM, OBS


def main():
    # buyer flag
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, choices=DELTA)
    parser.add_argument('--part', type=str, choices=HOLDOUT_PARTITIONS)
    parser.add_argument('--byr', action='store_true')
    args = parser.parse_args()

    # preliminaries
    log_dir = get_log_dir(byr=args.byr, delta=args.delta)
    values = load_values(part=args.part, delta=args.delta)
    df = pd.DataFrame(columns=['norm', 'price'])  # for output

    # rewards from data
    for dset in [OBS, SIM]:
        data = load_data(part=args.part, sim=(dset == SIM))
        df.loc[dset, :] = get_reward(
            data=data,
            values=values,
            delta=args.delta,
            byr=args.byr
        )

    # rewards from agent runs
    run_ids = [p for p in os.listdir(log_dir) if os.path.isdir(log_dir + p)]
    for run_id in run_ids:
        folder = log_dir + '{}/'.format(run_id)
        data = load_data(part=args.part, folder=folder)
        df.loc[run_id, :] = get_reward(
            data=data,
            values=values,
            delta=args.delta,
            byr=args.byr
        )

    # save table
    print(df)
    topickle(df, log_dir + '{}.pkl'.format(args.part))


if __name__ == '__main__':
    main()
