"""
Chunks a partition into SIM_CHUNKS pieces for reward generation
"""
import os
import numpy as np, pandas as pd
from compress_pickle import dump, load
from constants import PARTS_DIR, PARTITIONS
from rlenv.env_consts import LOOKUP_FILENAME, START_PRICE
from rlenv.env_utils import get_env_sim_subdir
from utils import init_x, input_partition
from constants import SIM_CHUNKS


def main():
    """
    Chunks the given partition
    """
    part = input_partition()

    # load inputs
    x_lstg = load('{}{}/x_lstg.gz')
    lookup = load('{}{}/{}'.format(PARTS_DIR, part, LOOKUP_FILENAME))

    # sort and subset
    lookup = lookup.drop('cat', axis=1).sort_values(by=START_PRICE)
    x_lstg = pd.concat([df.reindex(index=lookup.index) \
        for df in x_lstg.values()], axis=1)
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
        d = {'lookup': lookup.iloc[idx, :], 
             'x_lstg': x_lstg.iloc[idx, :]}
        dump(d, '{}{}.gz'.format(chunk_dir, i+1))
        # increment indices
        idx = idx + 1
        if idx[-1] >= len(x_lstg):
            idx = idx[:-1]


if __name__ == '__main__':
    main()
