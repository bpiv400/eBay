"""
Runs a simulation for the given partition and chunk of lstgs to generate
value estimations or discriminator inputs
"""
import argparse
import os
import torch
from constants import PARTITIONS
from rlenv.env_utils import get_env_sim_dir, get_env_sim_subdir, get_done_file
from rlenv.env_consts import NUM_CHUNKS
from rlenv.simulator.values.ValueGenerator import ValueGenerator
from rlenv.simulator.discrim.DiscrimGenerator import DiscrimGenerator


def chunk_done(part, num, values):
    """
    checks whether the intended outputs of the simulation have already been
    generated
    :param str part: partition in PARTITIONS
    :param int num: chunk number
    :param bool values: whether this is a value simulation
    :return: bool
    """
    records_dir = get_env_sim_subdir(part, values=values, discrim=not values)
    path = get_done_file(records_dir, num)
    return os.path.isfile(path)


def main():
    value_help = 'flag for whether to estimate values (generates discrim inputs'
    value_help = '{} if not)'.format(value_help)
    torch.set_default_dtype(torch.float32)

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, type=int, help='chunk number')
    parser.add_argument('--part', required=True, type=str, help='partition name')
    parser.add_argument('--values', action='store_true', help=value_help)

    # argument processing
    args = parser.parse_args()
    part = args.part
    num = args.num
    num = (num % NUM_CHUNKS)
    num = NUM_CHUNKS if num == 0 else num
    values = args.values
    # error checking
    if part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))

    # exit script if chunk has been generated
    if chunk_done(part, num, values):
        print('{} already done'.format(num))
        exit(0)

    # create value generate and generate outputs
    base_dir = get_env_sim_dir(part)
    if values:
        generator = ValueGenerator(base_dir, num)
    else:
        generator = DiscrimGenerator(base_dir, num)
    generator.generate()


if __name__ == '__main__':
    main()