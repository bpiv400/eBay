"""
Chunks a partition into NUM_CHUNKS pieces for value estimation
or generating discrim inputs
"""
import os
import numpy as np
import pandas as pd
from processing.processing_utils import load_file
from constants import SIM_CHUNKS
from featnames import START_PRICE, CAT
from rlenv.env_utils import get_env_sim_subdir, dump_chunk
from processing.processing_utils import input_partition


def main():
    part = input_partition()
    # load inputs
    x_lstg = load_file(part, 'x_lstg')
    lookup = load_file(part, 'lookup')

    # sort by start_price
    lookup = lookup.drop(CAT, axis=1).sort_values(by=START_PRICE)
    x_lstg = {k: v.reindex(index=lookup.index) for k, v in x_lstg.items()}

    # concatenate into one dataframe
    x_lstg = pd.concat(x_lstg.values(), axis=1)
    assert x_lstg.isna().sum().sum() == 0

    # make directories partition and components if they don't exist
    chunks = get_env_sim_subdir(part, chunks=True)
    values = get_env_sim_subdir(part, values=True)
    discrim = get_env_sim_subdir(part, discrim=True)
    for direct in [chunks, values, discrim]:
        if not os.path.isdir(direct):
            os.mkdir(direct)

    # iteration prep
    idx = np.arange(0, len(x_lstg), step=SIM_CHUNKS)
    chunk_dir = get_env_sim_subdir(part, chunks=True)

    # create chunks
    for i in range(SIM_CHUNKS):
        if (i+1) % 100 == 0:
            print('Chunk {} of {}'.format(i+1, SIM_CHUNKS))
        # create chunk and save
        lookup_chunk = lookup.iloc[idx, :]
        x_lstg_chunk = x_lstg.iloc[idx, :]
        path = '{}{}.gz'.format(chunk_dir, i+1)
        dump_chunk(x_lstg=x_lstg_chunk, lookup=lookup_chunk, path=path)
        # increment indices
        idx = idx + 1
        if idx[-1] >= len(x_lstg):
            idx = idx[:-1]


if __name__ == '__main__':
    main()
