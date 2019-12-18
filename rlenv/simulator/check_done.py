import argparse
import os
from constants import PARTITIONS
from rlenv.env_consts import NUM_CHUNKS
from rlenv.env_utils import get_env_sim_subdir

NUM_CHUNKS = 3

def check_done(part, values):
    subdir = get_env_sim_subdir(part, values=values, discrim=not values)
    for chunk in range(NUM_CHUNKS):
        chunk_path = '{}done_{}.txt'.format(subdir, chunk + 1)
        if not os.path.isfile(chunk_path):
            exit(0)
    exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--values', action='store_true',
                        help='flag for whether to check value creation')
    parser.add_argument('--part', type=str, help='one of {}'.format(PARTITIONS))
    args = parser.parse_args()
    print('part: {}'.format(args.part))
    print('vals: {}'.format(args.values))
    if args.part not in PARTITIONS:
        raise RuntimeError()
    check_done(args.part, args.values)


if __name__ == '__main__':
    main()
