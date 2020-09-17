import os
import pandas as pd
from agent.util import get_log_dir, find_best_run, get_valid_byr, get_value_byr
from utils import unpickle, topickle, load_data
from constants import VALIDATION, RL_BYR

PARTS = [VALIDATION, RL_BYR]


def main():
    # initialize output dataframe
    log_dir = get_log_dir(byr=True)
    run_ids = [p for p in os.listdir(log_dir) if os.path.isdir(log_dir + p)]
    idx = pd.MultiIndex.from_product([PARTS, run_ids], names=['part', 'run_id'])
    df = pd.DataFrame(columns=['norm', 'price'], index=idx).sort_index()

    # loop over partitions and runs
    best_run_dir = find_best_run(byr=False)
    for part in PARTS:
        values = unpickle(best_run_dir + '{}/values.pkl'.format(part))
        for run_id in run_ids:
            run_dir = log_dir + '{}/'.format(run_id)
            data = get_valid_byr(load_data(part=part, folder=run_dir))
            row = get_value_byr(data=data, values=values)
            if row is not None:
                df.loc[(part, run_id), :] = row

    # save table
    print(df)
    topickle(df, log_dir + 'runs.pkl')


if __name__ == '__main__':
    main()
