"""
prepare.py

Prepares input file for the environment in relevant partition subdirectory
"""
import pickle
import compress_pickle as cp
import h5py
from utils import cat_x_lstg
from rlenv.env_consts import (PARTITION, DATA_DIR, LOOKUP_FILENAME,
                              X_LSTG_FILENAME, X_LSTG, LOOKUP, LOOKUP_COLS_FILENAME,
                              ACC_PRICE, DEC_PRICE, START_PRICE)


def main():
    x_lstg = cat_x_lstg(PARTITION)
    x_lstg = x_lstg.to_numpy().astype('float64')

    lookup = cp.load('{}{}'.format(DATA_DIR, LOOKUP_FILENAME))
    lookup.reset_index(drop=False, inplace=True)
    cols = {col: i for i, col in enumerate(list(lookup.columns))}
    pickle.dump(cols, open(LOOKUP_COLS_FILENAME, 'wb'))
    lookup[ACC_PRICE] = lookup[ACC_PRICE] / lookup[START_PRICE]
    lookup[DEC_PRICE] = lookup[DEC_PRICE] / lookup[START_PRICE]
    lookup = lookup.to_numpy().astype('float64')

    f = h5py.File(X_LSTG_FILENAME, 'w')
    f.create_dataset(X_LSTG, shape=x_lstg.shape, dtype='float64', data=x_lstg)
    f.create_dataset(LOOKUP, shape=lookup.shape, dtype='float64', data=lookup)
    f.close()


if __name__ == '__main__':
    main()
