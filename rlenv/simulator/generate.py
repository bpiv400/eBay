"""
Generates simulator for a chunk of the lstgs in a partition
"""
import argparse
import os
import torch
from constants import PARTITIONS
from rlenv.env_utils import get_env_sim_dir, get_env_sim_subdir, get_done_file
from rlenv.simulator.values.ValueGenerator import ValueGenerator
from rlenv.simulator.discrim.DiscrimGenerator import DiscrimGenerator


def chunk_done(part, num, values):
    records_dir = get_env_sim_subdir(part, values=values, discrim=not values)
    path = get_done_file(records_dir, num)
    return os.path.isfile(path)


def main():
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, help='chunk number')
    parser.add_argument('--part', required=True, help='partition name')
    value_help = 'flag for whether to estimate values (generates discrim inputs'
    value_help = '{} if not)'.format(value_help)
    parser.add_argument('--values', action='store_true', help=value_help)

    args = parser.parse_args()
    part = args.part
    num = args.num
    values = args.values

    if part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))
    base_dir = get_env_sim_dir(part)

    if chunk_done(part, num, values):
        print('{} already done'.format(num))
        exit(0)

    if values:
        gen_class = ValueGenerator
    else:
        gen_class = DiscrimGenerator
    generator = gen_class(base_dir, num)
    generator.generate()


if __name__ == '__main__':
    main()