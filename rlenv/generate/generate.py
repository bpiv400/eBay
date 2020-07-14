from multiprocessing import Pool
import psutil
import argparse
import torch
from constants import PARTITIONS, NUM_CHUNKS
from rlenv.test.TestGenerator import TestGenerator
from agent.eval.ValueGenerator import ValueGenerator
from rlenv.generate.Generator import DiscrimGenerator


def main():
    torch.set_default_dtype(torch.float32)
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True, type=str)
    parser.add_argument('--name', required=True,
                        choices=['discrim', 'values', 'test'])
    parser.add_argument('--verbose', action='store_true',
                        help='print event detail')
    args = parser.parse_args()

    assert args.part in PARTITIONS
    print('{}: {}'.format(args.part, args.name))

    # create generator
    if args.name == 'values':
        cls = ValueGenerator
    elif args.name == 'test':
        cls = TestGenerator
    else:
        cls = DiscrimGenerator
    gen = cls(part=args.part, verbose=args.verbose)

    # process chunks in parallel
    num_workers = min(NUM_CHUNKS, psutil.cpu_count() - 1)
    pool = Pool(num_workers)
    pool.map(gen.process_chunk, list(range(NUM_CHUNKS)))


if __name__ == '__main__':
    main()
