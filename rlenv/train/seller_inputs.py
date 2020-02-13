import argparse
import os
import h5py
import numpy as np
from compress_pickle import load
from constants import PARTITIONS, ENV_SIM_DIR, PARTS_DIR, REINFORCE_INPUT_DIR
from rlenv.env_consts import X_LSTG_FILENAME, LOOKUP_FILENAME, X_LSTG, LOOKUP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True,
                        help='partition to chunk: {}'.format(PARTITIONS))
    part = parser.parse_args().part
    base_dir = '{}{}/'.format(ENV_SIM_DIR, part)
    reward_dir = '{}rewards'.format(base_dir)
    chunks = [path for path in os.listdir(reward_dir) if os.path.isdir(path)]
    lookup = load('{}{}/{}'.format(PARTS_DIR, part, LOOKUP_FILENAME))
    x_lstg = load('{}{}/{}'.format(PARTS_DIR, part, X_LSTG_FILENAME))
    path = '{}{}.gz'.format(REINFORCE_INPUT_DIR, part)
    store_inputs(x_lstg, lookup, path)


def store_inputs(x_lstg, lookup, path):
    lookup = lookup.reset_index(drop=False)
    lookup_cols = [col.encode('utf-8') for col in list(lookup.columns)]
    f = h5py.File(path)
    lookup_vals = lookup.values.astype(np.float32)
    x_lstg_vals = x_lstg.values.astype(np.float32)
    lookup = f.create_dataset(LOOKUP, lookup_vals)
    f.create_dataset(X_LSTG, x_lstg_vals)
    lookup.attrs['cols'] = lookup_cols
    f.close()


if __name__ == '__main__':
    main()
