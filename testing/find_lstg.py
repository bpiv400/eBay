"""
Finds an arbitrary lstg given partition
"""
import argparse
import numpy as np
from constants import NUM_CHUNKS
from featnames import VALIDATION, PARTITIONS
from rlenv.util import load_chunk, get_env_sim_dir
from utils import compose_args


def main():
    parser = argparse.ArgumentParser()
    args = {
        'lstg': {'required': True, 'type': int},
        'part': {'default': VALIDATION, 'choices': PARTITIONS}
    }
    compose_args(parser=parser, arg_dict=args)
    args = parser.parse_args()
    part, lstg = args.part, args.lstg

    base_dir = get_env_sim_dir(part)
    for i in range(NUM_CHUNKS):
        _, lookup, _ = load_chunk(base_dir=base_dir, num=i)
        lstgs = lookup.index
        if lstg in lstgs:
            print('Found lstg in {} chunk {}: index {} of {}'.format(
                part, i, np.where(lstgs == lstg)[0][0] + 1, len(lstgs)))
            exit(0)


if __name__ == '__main__':
    main()
