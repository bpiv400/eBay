"""
Chunks a partition into NUM_CHUNKS pieces for value estimation
or generating discrim inputs
"""
import os
import h5py
import numpy as np
import pandas as pd
from compress_pickle import dump
from agent.agent_consts import SELLER_TRAIN_INPUT
from utils import load_file, input_partition, init_x
from constants import SIM_CHUNKS, PARTS_DIR, TRAIN_RL
from featnames import START_PRICE, CAT
from rlenv.env_consts import X_LSTG, LOOKUP

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def store_inputs(x_lstg, lookup, path):
    lookup = lookup.reset_index(drop=False)
    lookup_cols = [col.encode('utf-8') for col in list(lookup.columns)]
    f = h5py.File(path, 'w')
    lookup_vals = lookup.values.astype(np.float32)
    x_lstg_vals = x_lstg.values.astype(np.float32)
    lookup = f.create_dataset(LOOKUP, data=lookup_vals)
    f.create_dataset(X_LSTG, data=x_lstg_vals)
    lookup.attrs['cols'] = lookup_cols
    f.close()


def main():
    # partition from command line
    part = input_partition()

    # load inputs
    lookup = load_file(part, LOOKUP).drop(CAT, axis=1).sort_values(
        by=START_PRICE)
    x_lstg = init_x(part, lookup.index)

    # concatenate into one dataframe
    x_lstg = pd.concat(x_lstg.values(), axis=1)
    assert x_lstg.isna().sum().sum() == 0

    # save RL seller input as HDF5 file
    if part == TRAIN_RL:
        store_inputs(x_lstg, lookup, SELLER_TRAIN_INPUT)

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
