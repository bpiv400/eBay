"""
Runs a simulation for the given partition and chunk of lstgs to generate
value estimations or discriminator inputs
"""
import os, argparse
import torch
from constants import PARTITIONS, SIM_CHUNKS
from rlenv.env_utils import get_env_sim_dir
from rlenv.simulator.values.ValueGenerator import ValueGenerator
from rlenv.simulator.Generator import Generator
from rlenv.simulator.discrim.DiscrimGenerator import DiscrimGenerator
from rlenv.test.TestGenerator import TestGenerator


def chunk_done(generator):
    """
    checks whether the intended outputs of the simulation have already been
    generated
    :param ValueGenerator generator: generator object for the chunk / exp type
    :return: bool
    """
    records_path = generator.records_path
    return os.path.isfile(records_path)


def get_gen_class(values=False, test=False):
    if values:
        return ValueGenerator
    elif test:
        return TestGenerator
    else:
        return DiscrimGenerator


def main():
    torch.set_default_dtype(torch.float32)
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, type=int, help='chunk number')
    parser.add_argument('--part', required=True, type=str, help='partition name')
    parser.add_argument('--values', action='store_true',
                        help='for value estimation; otherwise discriminator input')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--verbose', action='store_true',
                        help='print event detail')
    parser.add_argument('--thread', required=False, type=int)

    args = parser.parse_args()
    num, part, values, verbose = args.num, args.part, args.values, args.verbose
    if part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))
    print('{} {}: {}'.format(part, num, 'values' if values else 'discrim'))
    
    # num is modulus SIM_CHUNKS
    num = ((num-1) % SIM_CHUNKS) + 1

    # check whether chunk is finished processing

    # create generator
    gen_class = get_gen_class(values=values, test=args.test)
    generator = gen_class(direct=get_env_sim_dir(part), verbose=verbose, start=args.thread)   # type: Generator
    generator.load_chunk(chunk=num)
    generator.initialize()
    if values and chunk_done(generator):
        print('{} already done'.format(num))
        exit(0)
    generator.generate()


if __name__ == '__main__':
    print('main')
    main()
