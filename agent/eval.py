import argparse
import os
import pandas as pd
from agent.util import get_log_dir, get_reward, load_values
from utils import topickle, load_data, compose_args
from agent.const import AGENT_PARAMS
from constants import AGENT_PARTITIONS, TEST
from featnames import SIM, OBS, X_OFFER


def main():
    # buyer flag
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    args = parser.parse_args()

    # preliminaries
    log_dir = get_log_dir(byr=args.byr, delta=args.delta)
    values = load_values(part=args.part, delta=args.delta)
    df = pd.DataFrame(columns=['norm', 'price'])  # for output

    # rewards from data
    if not args.byr:
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
        run_dir = log_dir + '{}/'.format(run_id)
        data = load_data(part=args.part, run_dir=run_dir)
        if X_OFFER in data:
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
