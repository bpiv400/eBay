"""
Runs a simulation for the given partition and chunk of lstgs to generate
value estimations or discriminator inputs
"""
import os
import argparse
import torch
from constants import PARTITIONS, PARTS_DIR
from sim.values.ValueGenerator import ValueGenerator
from rlenv import Generator
from sim.outcomes import OutcomeGenerator
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


def get_gen_class(name):
    if name == 'values':
        return ValueGenerator
    elif name == 'test':
        return TestGenerator
    else:
        return OutcomeGenerator


def main():
    torch.set_default_dtype(torch.float32)
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, type=int, help='chunk number')
    parser.add_argument('--part', required=True, type=str, help='partition name')
    parser.add_argument('--name', choices=['outcomes', 'values', 'test'], default='outcomes')
    parser.add_argument('--verbose', action='store_true',
                        help='print event detail')
    parser.add_argument('--thread', required=False, type=int)
    args = parser.parse_args()

    assert args.part in PARTITIONS
    print('{} {}: {}'.format(args.part, args.num, args.name))

    # directory for processed chunks
    chunk_dir = PARTS_DIR + '{}/{}/'.format(args.part, args.name)
    if not os.path.isdir(chunk_dir):
        os.mkdir(chunk_dir)

    # create generator
    gen_class = get_gen_class(args.name)
    generator = gen_class(direct=chunk_dir,
                          verbose=args.verbose,
                          start=args.thread)   # type: Generator
    if args.values and chunk_done(args.num):
        print('{} already done'.format(args.num))
        exit(0)
    generator.process_chunk(chunk=args.num)


if __name__ == '__main__':
    main()
