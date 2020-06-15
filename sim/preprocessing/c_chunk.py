"""
Chunks a partition into NUM_CHUNKS pieces for value estimation
or generating discrim inputs
"""
import os
import numpy as np
import pandas as pd
from compress_pickle import dump
from utils import load_file, input_partition, init_x
from constants import SIM_CHUNKS, PARTS_DIR
from featnames import START_PRICE, CAT
from rlenv.const import LOOKUP


def main():
    # partition from command line
    part = input_partition()

    # load inputs
    lookup = load_file(part, LOOKUP)..sort_values(by=START_PRICE)
    x_lstg = init_x(part, lookup.index)

    # concatenate into one dataframe
    x_lstg = pd.concat(x_lstg.values(), axis=1)
    assert x_lstg.isna().sum().sum() == 0

    # make chunk directory
    chunk_dir = PARTS_DIR + '{}/chunks/'.format(part)
    if not os.path.isdir(chunk_dir):
        os.mkdir(chunk_dir)

    # iteration prep
    idx = np.arange(0, len(x_lstg), step=SIM_CHUNKS)

    # create chunks
    for i in range(SIM_CHUNKS):
        if (i+1) % 100 == 0:
            print('Chunk {} of {}'.format(i+1, SIM_CHUNKS))

        # create chunk and save
        chunk = {'lookup': lookup.iloc[idx, :],
                 'x_lstg': x_lstg.iloc[idx, :]}
        path = chunk_dir + '{}.gz'.format(i+1)
        dump(chunk, path)

        # increment indices
        idx = idx + 1
        if idx[-1] >= len(x_lstg):
            idx = idx[:-1]


if __name__ == '__main__':
    main()
