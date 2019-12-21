"""
Generates simulator for a chunk of the lstgs in a partition
"""
import os, argparse
import torch
from constants import PARTITIONS
from rlenv.env_utils import get_env_sim_dir, get_env_sim_subdir, get_done_file
from rlenv.env_consts import NUM_CHUNKS
from rlenv.simulator.values.ValueGenerator import ValueGenerator
from rlenv.simulator.discrim.DiscrimGenerator import DiscrimGenerator


def chunk_done(part, num, values):
    records_dir = get_env_sim_subdir(part, values=values, discrim=not values)
    path = get_done_file(records_dir, num)
    return os.path.isfile(path)


def main():
    torch.set_default_dtype(torch.float32)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, type=int, help='chunk number')
    parser.add_argument('--part', required=True, type=str, help='partition name')
    parser.add_argument('--values', action='store_true', 
        help='for value estimation; otherwise discriminator input')
    parser.add_argument('--verbose', action='store_true',
        help='print event detail')
    args = parser.parse_args()
    num, part, values, verbose = args.num, args.part, args.values, args.verbose
    if part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))
    print('{} {}: {}'.format(part, num, 'values' if values else 'discrim'))
    
    # num is modulus NUM_CHUNKS
    num = ((num-1) % NUM_CHUNKS) + 1

    # check whether chunk is finished processing
    if chunk_done(part, num, values):
        print('{} already done'.format(num))
        exit(0)

    # create generator
    gen_class = ValueGenerator if values else DiscrimGenerator
    generator = gen_class(get_env_sim_dir(part), num, verbose)
    generator.generate()


if __name__ == '__main__':
    main()