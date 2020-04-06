"""
Chunks a partition into NUM_CHUNKS pieces for value estimation
or generating discrim inputs
"""
import os
import numpy as np
from compress_pickle import load
from constants import PARTS_DIR, SIM_CHUNKS
from featnames import START_PRICE
from rlenv.env_consts import LOOKUP_FILENAME
from rlenv.env_utils import get_env_sim_subdir, align_x_lstg_lookup, dump_chunk
from processing.processing_utils import input_partition


def main():
    part = input_partition()
    # load inputs
    x_lstg = load('{}{}/x_lstg.gz'.format(PARTS_DIR, part))
    lookup = load('{}{}/{}'.format(PARTS_DIR, part, LOOKUP_FILENAME))

    # sort and subset
    lookup = lookup.drop('cat', axis=1).sort_values(by=START_PRICE)
    x_lstg = align_x_lstg_lookup(x_lstg, lookup)
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
