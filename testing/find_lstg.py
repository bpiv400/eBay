"""
Finds an arbitrary lstg given partition
"""
import argparse
from constants import VALIDATION, PARTITIONS, NUM_RL_WORKERS
from rlenv.util import load_chunk, get_env_sim_dir
from utils import compose_args


def main():
    parser = argparse.ArgumentParser()
    args = {
        'lstg': {'required': True,
                 'type': int},
        'part': {'required': True,
                 'default': VALIDATION,
                 'choices': PARTITIONS}
    }
    compose_args(parser=parser, arg_dict=args)
    args = parser.parse_args()
    base_dir = get_env_sim_dir(args.part)
    for i in range(NUM_RL_WORKERS):
        _, lookup, _ = load_chunk(base_dir=base_dir, num=i)
        if args.lstg in lookup.index:
            print('Found lstg in {} chunk {}'.format(args.part,
                                                     i))
            exit(0)


if __name__ == '__main__':
    main()
