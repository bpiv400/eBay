"""
Runs a simulation for the given partition and chunk of lstgs to generate
value estimations or discriminator inputs
"""
import os
import argparse
import torch
from constants import PARTITIONS
from rlenv.Generator import Generator
from testing.TestGenerator import TestGenerator
from sim.values.ValueGenerator import ValueGenerator
from sim.outcomes.OutcomeGenerator import OutcomeGenerator


def chunk_done(generator):
    """
    checks whether the intended outputs of the simulation have already been
    generated
    :param ValueGenerator generator: generator object for the chunk / exp type
    :return: bool
    """
    records_path = generator.records_path
    return os.path.isfile(records_path)


def get_generator_class(name):
    if name == 'values':
        return ValueGenerator
    elif name == 'testing':
        return TestGenerator
    else:
        return OutcomeGenerator


def get_generator_args(args):
    kwargs = dict(part=args.part,
                  verbose=args.verbose)
    if args.thread is not None:
        kwargs['thread'] = args.thread
    return kwargs


def main():
    torch.set_default_dtype(torch.float32)
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, type=int, help='chunk number')
    parser.add_argument('--part', required=True, type=str, help='partition name')
    parser.add_argument('--name', choices=['outcomes', 'values', 'testing'], default='outcomes')
    parser.add_argument('--verbose', action='store_true',
                        help='print event detail')
    parser.add_argument('--thread', required=False, type=int)
    args = parser.parse_args()

    assert args.part in PARTITIONS
    print('{} {}: {}'.format(args.part, args.num, args.name))

    # create generator
    gen_args = get_generator_args(args)
    gen_class = get_generator_class(args.name)
    generator = gen_class(**gen_args)   # type: Generator
    if args.name == 'values' and chunk_done(args.num):
        print('{} already done'.format(args.num))
        exit(0)
    generator.process_chunk(chunk=args.num)


if __name__ == '__main__':
    main()
