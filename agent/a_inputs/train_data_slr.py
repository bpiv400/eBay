"""
Creates hdf5 file containing input (lookup and x_lstg) for training seller rl

By default, uses the train_rl partition
"""
import os
import argparse
import h5py
import numpy as np
from compress_pickle import load
from featnames import CAT
from constants import PARTITIONS, PARTS_DIR, TRAIN_RL
from agent.agent_consts import SELLER_TRAIN_INPUT
from agent.a_inputs.inputs_utils import (remove_unlikely_arrival_lstgs,
                                         add_no_arrival_likelihood)
from rlenv.env_consts import X_LSTG_FILENAME, LOOKUP_FILENAME, X_LSTG, LOOKUP
from rlenv.env_utils import align_x_lstg_lookup


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=False,
                        help='partition to chunk: {}'.format(PARTITIONS))
    parser.add_argument('--remove', action='store_true', default=False)
    args = parser.parse_args()
    part = args.part
    remove = args.remove
    if part is None:
        part = TRAIN_RL
    lookup = load('{}{}/{}'.format(PARTS_DIR, part, LOOKUP_FILENAME))
    lookup = lookup.drop(columns=[CAT])
    x_lstg = load('{}{}/{}'.format(PARTS_DIR, part, X_LSTG_FILENAME))
    x_lstg = align_x_lstg_lookup(x_lstg=x_lstg, lookup=lookup)
    if remove:
        x_lstg, lookup = remove_unlikely_arrival_lstgs(x_lstg=x_lstg,
                                                       lookup=lookup)
    else:
        lookup = add_no_arrival_likelihood(x_lstg=x_lstg, lookup=lookup)
    store_inputs(x_lstg, lookup, SELLER_TRAIN_INPUT)


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


if __name__ == '__main__':
    main()
