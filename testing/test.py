import argparse
from testing.const import SCRIPT_PARAMS, TEST_GENERATOR_KWARGS
from testing.TestGenerator import TestGenerator
from utils import compose_args


def main():
    parser = argparse.ArgumentParser()
    compose_args(arg_dict=SCRIPT_PARAMS, parser=parser)
    args = vars(parser.parse_args())
    generator_kwargs = dict()
    for arg in TEST_GENERATOR_KWARGS:
        generator_kwargs[arg] = args[arg]
    gen = TestGenerator(**generator_kwargs)
    gen.process_chunk(chunk=args['num'])


if __name__ == '__main__':
    main()
