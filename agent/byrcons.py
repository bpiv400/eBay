from shutil import rmtree, move
import numpy as np
import pandas as pd
from agent.util import get_log_dir
from agent.const import BYR_CONS, NEW_BYR_CONS


def main():
    # map old indices to new indices
    idx = pd.Series(-1, index=BYR_CONS.index, dtype=int)
    for i in BYR_CONS.index:
        row = BYR_CONS.loc[i].to_list()
        if row[1] == 1:
            if row[2] == .2:
                row[2] = -1
            else:
                continue
        idx.loc[i] = np.argwhere((row == NEW_BYR_CONS.values).all(axis=1))[0][0]

    # delete folders
    for i in idx.index:
        folder = get_log_dir(byr=True) + 'heuristic/{}/'.format(i)
        if idx.loc[i] == -1:
            print('Removing {}'.format(folder))
            rmtree(folder)

    # move folders
    for i in reversed(list(idx.index)):
        new_i = idx.loc[i]
        if new_i > -1:
            folder = get_log_dir(byr=True) + 'heuristic/{}/'.format(i)
            new_folder = get_log_dir(byr=True) + 'heuristic/{}/'.format(new_i)
            print('Moving {} to {}'.format(i, new_i))
            move(folder, new_folder)


if __name__ == '__main__':
    main()
