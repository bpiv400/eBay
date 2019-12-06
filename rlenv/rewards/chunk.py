"""
Chunks a partition into NUM_CHUNKS pieces for reward generation

"""
import argparse
from compress_pickle import dump, load
from constants import PARTS_DIR, REWARDS_DIR, PARTITIONS
from rlenv.env_consts import LOOKUP_FILENAME, NUM_CHUNKS


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
    x_lstg = load('{}{}/x_lstg.gz'.format(PARTS_DIR, part))
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
        path = '{}{}/chunks/{}.gz'.format(REWARDS_DIR, part, (i + 1))
        curr_dict = {
            'lookup': curr_lookup,
            'x_lstg': curr_df
        }
        dump(curr_dict, path)


if __name__ == '__main__':
    main()
