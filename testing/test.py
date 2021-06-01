import argparse
from testing.TestGenerator import TestGenerator
from testing.agents.AgentTestGenerator import AgentTestGenerator
from constants import NUM_CHUNKS
from featnames import VALIDATION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--slr', action='store_true')
    parser.add_argument('--num', type=int, default=1,
                        choices=range(1, NUM_CHUNKS+1))
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.byr or args.slr:
        assert not (args.slr and args.byr)
        gen = AgentTestGenerator(verbose=args.verbose,
                                 byr=args.byr,
                                 slr=args.slr)
    else:
        gen = TestGenerator(verbose=args.verbose)
    gen.process_chunk(part=VALIDATION, chunk=args.num-1)


if __name__ == '__main__':
    main()
