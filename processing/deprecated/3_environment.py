"""
Prepares hdf5 file for reinforcement learning training input

Expects to be run from project root
"""
import pickle
import os
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from env_constants import LSTG_COLUMNS

# paths for inputs and outputs
H5_PATH = 'data/rl/input.h5'
CHUNK_PATH = 'data/chunks'


def save_slr(data, file=None):
    """
    Saves slr lstgs to new data_set in h5py file

    :param data: dataframe containing 1 entry for each listing
    :param file: h5py.File object giving output file
    :return: NA
    """
    slr = data['slr'].iloc[0]
    assert (isinstance(data, pd.DataFrame))
    subset = data.loc[:, LSTG_COLUMNS]
    subset = subset.to_numpy()
    try:
        file.create_dataset(str(slr), data=subset)
    except RuntimeError:
        pass


def main():
    """
    Main method
    :return: NA
    """
    files = os.listdir(CHUNK_PATH)
    files = [file for file in files if 'out' in file]
    output = h5py.File(H5_PATH, 'w')
    all_columns = ['slr'] + LSTG_COLUMNS
    slrs = []
    for file in files:
        chunk = pickle.load(open('%s/%s' % (CHUNK_PATH, file), 'rb'))['T']
        print(chunk.columns)
        print('na: {}'.format(chunk['end_date'].isna().any()))
        chunk.reset_index(drop=False, inplace=True)
        chunk.drop_duplicates(subset=['slr', 'lstg'], keep='first', inplace=True)
        chunk.drop(columns=['byr_us', 'byr_hist', 'thread'], inplace=True)
        chunk = chunk.reindex(all_columns, axis=1, copy=False)
        tqdm.pandas()
        slr_groups = chunk.groupby(by='slr', sort=False, as_index=False)
        slr_groups.progress_apply(save_slr, file=output)
        slrs += chunk['slr'].unique().tolist()
    assert (len(output) == len(slrs))
    slrs = np.array(slrs, dtype=np.int64)
    output.create_dataset('slrs', data=slrs)
    output.close()


if __name__ == '__main__':
    main()
