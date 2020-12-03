import argparse
from testing.TestGenerator import TestGenerator
from testing.agents.AgentTestGenerator import AgentTestGenerator
from utils import compose_args
from featnames import VALIDATION

SCRIPT_PARAMS = {'byr': {'action': 'store_true'},
                 'slr': {'action': 'store_true'},
                 'num': {'type': int, 'default': 0},
                 'verbose': {'action': 'store_true'}
                 }


def main():
    parser = argparse.ArgumentParser()
    compose_args(arg_dict=SCRIPT_PARAMS, parser=parser)
    args = parser.parse_args()
    if args.byr or args.slr:
        assert not (args.slr and args.byr)
        gen = AgentTestGenerator(verbose=args.verbose, byr=args.byr, slr=args.slr)
    else:
        gen = TestGenerator(verbose=args.verbose)
    gen.process_chunk(part=VALIDATION, chunk=args.num)


if __name__ == '__main__':
    main()
