"""
Chunks a partition into NUM_CHUNKS pieces for reward generation

"""
import os
import argparse
import pandas as pd
import numpy as np
from compress_pickle import dump, load
from constants import PARTS_DIR, PARTITIONS
from rlenv.env_consts import LOOKUP_FILENAME, NUM_CHUNKS, START_PRICE
from rlenv.env_utils import get_env_sim_subdir
import utils


def init_x(part):
    x = utils.init_x(part, None)
    x = [frame for _, frame in x.items()]
    x = pd.concat(x, axis=1)
    return x


def make_dirs(part):
    """
    Create subdirectories for this partition of the environment simulation
    :param str part: one of PARTITIONS
    :return: None
    """
    chunks = get_env_sim_subdir(part, chunks=True)
    values = get_env_sim_subdir(part, values=True)
    discrim = get_env_sim_subdir(part, discrim=True)
    for direct in [chunks, values, discrim]:
        if not os.path.isdir(direct):
            os.mkdir(direct)


def sort_inputs(x_lstg, lookup):
    x_lstg = x_lstg.sort_values(by=START_PRICE)
    lookup = lookup.reindex(x_lstg.index)
    return x_lstg, lookup


def main():
    """
    Chunks the given partition
    """
    # prep
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True,
                        help='partition to chunk: {}'.format(PARTITIONS))
    part = parser.parse_args().part
    if part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))
    # load inputs
    x_lstg = init_x(part)
    lookup = load('{}{}/{}'.format(PARTS_DIR, part, LOOKUP_FILENAME))
    # make directories partition and components if they don't exist
    make_dirs(part)
    # error checking and sorting
    assert (lookup.index == x_lstg.index).all()
    x_lstg, lookup = sort_inputs(x_lstg, lookup)
    # iteration prep
    total_lstgs = len(x_lstg)
    indices = np.arange(0, total_lstgs, step=NUM_CHUNKS)
    chunk_dir = get_env_sim_subdir(part, chunks=True)
    # create chunks
    for i in range(NUM_CHUNKS):
        indices = indices + 1
        if indices[-1] >= total_lstgs:
            indices = indices[:-1]
        curr_df = x_lstg.iloc[indices, :]
        curr_lookup = lookup.iloc[indices, :]
        path = '{}{}.gz'.format(chunk_dir, (i + 1))
        # store output
        curr_dict = {
            'lookup': curr_lookup,
            'x_lstg': curr_df
        }
        dump(curr_dict, path)


if __name__ == '__main__':
    main()
