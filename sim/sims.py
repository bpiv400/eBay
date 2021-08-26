import argparse
import os
from rlenv.generate.Generator import OutcomeGenerator, ValueGenerator
from utils import topickle
from constants import NUM_CHUNKS, ARRIVAL_SIMS, OUTCOME_SIMS
from paths import SIM_DIR
from featnames import AGENT_PARTITIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, required=True,
                        choices=AGENT_PARTITIONS)
    parser.add_argument('--num', type=int, required=True,
                        choices=range(1, NUM_CHUNKS + 1))
    parser.add_argument('--values', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # generator
    gen_cls = ValueGenerator if args.values else OutcomeGenerator

    # create output folder
    output_dir = SIM_DIR + '{}/{}/'.format(
        args.part, 'values' if args.values else 'outcomes')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # check if chunk has already been processed
    chunk = args.num - 1
    path = output_dir + '{}.pkl'.format(chunk)
    if os.path.isfile(path):
        print('Chunk {} already exists.'.format(chunk))
        exit(0)

    # process one chunk
    gen = gen_cls(verbose=args.verbose)
    obj = gen.process_chunk(part=args.part,
                            chunk=chunk,
                            num_sims=ARRIVAL_SIMS if args.values else OUTCOME_SIMS)

    # save
    topickle(obj, path)


if __name__ == '__main__':
    main()
