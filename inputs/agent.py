"""
Chunks a partition into NUM_CHUNKS pieces for value estimation
or generating discrim inputs
"""
import os
import h5py
import numpy as np
import pandas as pd
from utils import load_file, init_x
from constants import NO_ARRIVAL_CUTOFF, TRAIN_RL, RL_TRAIN_DIR, \
    NUM_WORKERS_RL
from featnames import CAT, START_PRICE
from rlenv.const import X_LSTG, LOOKUP

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def get_cols(df):
    return [c.encode('utf-8') for c in list(df.columns)]


def main():
    # load lookup
    lookup = load_file(TRAIN_RL, LOOKUP).drop(CAT, axis=1)

    # drop listings with infrequent arrivals
    lookup = lookup[lookup.p_no_arrival < NO_ARRIVAL_CUTOFF]
    lookup = lookup.drop('p_no_arrival', axis=1)

    # sort by start_price
    lookup = lookup.sort_values(by=START_PRICE)
    idx = lookup.index
    lookup = lookup.reset_index(drop=False).astype('float32')

    # input features
    x_lstg = init_x(TRAIN_RL, idx)
    x_lstg = pd.concat(x_lstg.values(), axis=1).astype('float32')
    assert x_lstg.isna().sum().sum() == 0

    # iteration prep
    idx = np.arange(0, len(x_lstg), step=NUM_WORKERS_RL)

    # columns names
    lookup_cols = get_cols(lookup)
    x_lstg_cols = get_cols(x_lstg)

    # split and save as hdf5
    for i in range(NUM_WORKERS_RL):
        print('Chunk {} of {}'.format(i+1, NUM_WORKERS_RL))

        # split dataframes, convert to numpy
        lookup_i = lookup.iloc[idx, :].values
        x_lstg_i = x_lstg.iloc[idx, :].values

        # save to file
        f = h5py.File(RL_TRAIN_DIR + '{}.hdf5'.format(i), 'w')
        lookup_dataset = f.create_dataset(LOOKUP, data=lookup_i)
        x_lstg_dataset = f.create_dataset(X_LSTG, data=x_lstg_i)
        lookup_dataset.attrs['cols'] = lookup_cols
        x_lstg_dataset.attrs['cols'] = x_lstg_cols
        f.close()

        # increment indices
        idx = idx + 1
        if idx[-1] >= len(x_lstg):
            idx = idx[:-1]


if __name__ == '__main__':
    main()
