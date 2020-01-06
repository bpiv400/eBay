"""
Checks whether target outputs (i.e. discrim inputs or value estimates)
have already been generated for all chunks in the partition

Exits with error code if all have been finished
"""

import argparse
import os
from constants import PARTITIONS, SIM_CHUNKS
from rlenv.env_utils import get_env_sim_subdir, get_done_file


def check_done(part, values):
    """
    Checks whether all target outputs have already been generated
    :param str part: partition in PARTITIONS
    :param bool values: whether the target outputs are value estimates
    """
    subdir = get_env_sim_subdir(part, values=values, discrim=not values)
    for chunk in range(SIM_CHUNKS):
        chunk_path = get_done_file(subdir, chunk + 1)
        if not os.path.isfile(chunk_path):
            exit(0)
    print('got through all')
    exit(1)


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--values', action='store_true',
                        help='flag for whether to check value creation')
    parser.add_argument('--part', type=str, help='one of {}'.format(PARTITIONS))
    args = parser.parse_args()
    # error checking
    if args.part not in PARTITIONS:
        raise RuntimeError()
    check_done(args.part, args.values)


if __name__ == '__main__':
    main()
