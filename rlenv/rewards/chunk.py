"""
Chunks a partition into NUM_CHUNKS pieces for reward generation

"""
import argparse
from compress_pickle import dump, load
from processing.processing_utils import cat_x_lstg
from constants import PARTS_DIR, REWARDS_DIR, PARTITIONS
from rlenv.env_consts import LOOKUP_FILENAME

NUM_CHUNKS = 512


def path_func(part):
    """
    Creates function to generate paths to x_lstg components

    :return: function
    """
    prefix = '{}{}/'.format(PARTS_DIR, part)

    def part_path(comps):
        output = '{}{}_{}.gz'.format(prefix, comps[0], comps[1])
        return output

    return part_path


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
    x_lstg = cat_x_lstg(path_func(part))
    lookup = load('{}{}/{}'.format(PARTS_DIR, part, LOOKUP_FILENAME))
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
        path = '{}{}/{}.gz'.format(REWARDS_DIR, part, (i + 1))
        curr_dict = {
            'lookup': curr_lookup,
            'x_lstg': curr_df
        }
        dump(curr_dict, path)


if __name__ == '__main__':
    main()
