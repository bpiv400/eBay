"""
Chunks a partition into NUM_CHUNKS pieces for value estimation
or generating discrim inputs
"""
import os
import h5py
import pandas as pd
from agent.agent_consts import SELLER_TRAIN_INPUT
from utils import load_file, init_x
from constants import NO_ARRIVAL_CUTOFF, TRAIN_RL
from featnames import CAT
from rlenv.env_consts import X_LSTG, LOOKUP

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def get_cols(df):
    return [c.encode('utf-8') for c in list(df.columns)]


def main():
    # load lookup
    lookup = load_file(TRAIN_RL, LOOKUP).drop(CAT, axis=1)

    # drop listings with infrequent arrivals
    lookup = lookup[lookup.p_no_arrival < NO_ARRIVAL_CUTOFF]
    lookup = lookup.drop('p_no_arrival', axis=1)
    idx = lookup.index
    lookup = lookup.reset_index(drop=False)

    # input features
    x_lstg = init_x(TRAIN_RL, idx)
    x_lstg = pd.concat(x_lstg.values(), axis=1)
    assert x_lstg.isna().sum().sum() == 0

    # save hdf5 file
    f = h5py.File(SELLER_TRAIN_INPUT, 'w')
    lookup_dataset = f.create_dataset(LOOKUP,
                                      data=lookup.values.astype('float32'))
    x_lstg_dataset = f.create_dataset(X_LSTG,
                                      data=x_lstg.values.astype('float32'))
    lookup_dataset.attrs['cols'] = get_cols(lookup)
    x_lstg_dataset.attrs['cols'] = get_cols(x_lstg)
    f.close()


if __name__ == '__main__':
    main()
