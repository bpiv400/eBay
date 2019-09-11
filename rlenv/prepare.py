"""
prepare.py

Prepares input for environment by pulling lstg data out of
train_rl/x_lstg.pkl and storing it in a .hdf5 file called
x_lstg.hdf5.

x_lstg contains:
'lstg': matrix
Stores dictionary mapping column names to col position in lstg.hdf5

"""
import pandas
import pickle
import h5py
from utils import unpickle
from rlenv.env_constants import LSTG_FILENAME, COL_FILENAME
# constants
INPUT_PATH = 'data/partitions/train_rl/x_lstg.pkl'


def main():
    x = unpickle(INPUT_PATH)
    column_dict = dict()
    for i, col in enumerate(x.columns):
        column_dict[col] = i
    pickle.dump(column_dict, open(COL_FILENAME, 'wb'))
    f = h5py.File(LSTG_FILENAME, 'w')
    lstg = f.create_dataset('lstg', shape=x.shape, dtype='float64')
    x = x.to_numpy()
    x = x.astype('float32')
    print(x.dtype)
    lstg[:, :] = x
    f.close()


if __name__ == '__main__':
    main()
