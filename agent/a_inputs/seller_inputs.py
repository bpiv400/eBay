import os
import argparse
import h5py
import numpy as np
from compress_pickle import load
from utils import align_x_lstg_lookup
from featnames import CAT, START_PRICE, DEC_PRICE, ACC_PRICE
from constants import PARTITIONS, PARTS_DIR
from rlenv.env_consts import X_LSTG_FILENAME, LOOKUP_FILENAME, X_LSTG, LOOKUP
from agent.agent_utils import slr_input_path


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True,
                        help='partition to chunk: {}'.format(PARTITIONS))
    part = parser.parse_args().part
    lookup = load('{}{}/{}'.format(PARTS_DIR, part, LOOKUP_FILENAME))
    lookup = lookup.drop(columns=[CAT])
    lookup.loc[:, DEC_PRICE] = 0.0
    lookup.loc[:, ACC_PRICE] = lookup.loc[:, START_PRICE]
    x_lstg = load('{}{}/{}'.format(PARTS_DIR, part, X_LSTG_FILENAME))
    x_lstg = align_x_lstg_lookup(x_lstg, lookup)
    path = slr_input_path(part=part)
    store_inputs(x_lstg, lookup, path)


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
