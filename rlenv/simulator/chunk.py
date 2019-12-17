"""
Chunks a partition into NUM_CHUNKS pieces for reward generation

"""
import argparse
import pandas as pd
from compress_pickle import dump, load
from constants import PARTS_DIR, ENV_SIM_DIR, PARTITIONS
from rlenv.env_consts import LOOKUP_FILENAME, NUM_CHUNKS
from rlenv.env_utils import get_env_sim_dir, get_env_sim_subdir
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
    get_env_sim_subdir()



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
    x_lstg = init_x(part)
    lookup = load('{}{}/{}'.format(PARTS_DIR, part, LOOKUP_FILENAME))
    make_dirs(part)
    assert (lookup.index == x_lstg.index).all()
    total_lstgs = len(x_lstg)
    per_chunk = int(total_lstgs / NUM_CHUNKS)
    # create chunks
    for i in range(NUM_CHUNKS):
        start = i * per_chunk
        if i != NUM_CHUNKS - 1:
            end = (i + 1) * per_chunk
        else:
            end = total_lstgs
        # store chunk
        curr_df = x_lstg.iloc[start:end, :]
        curr_lookup = lookup.iloc[start:end, :]
        path = '{}{}/chunks/{}.gz'.format(ENV_SIM_DIR, part, (i + 1))
        curr_dict = {
            'lookup': curr_lookup,
            'x_lstg': curr_df
        }
        dump(curr_dict, path)


if __name__ == '__main__':
    main()
