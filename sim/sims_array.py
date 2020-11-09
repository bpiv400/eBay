import argparse
import os
from rlenv.generate.Generator import OutcomeGenerator, ValueGenerator
from sim.envs import SimulatorEnv
from utils import topickle
from constants import SIM_DIR, NUM_CHUNKS
from featnames import AGENT_PARTITIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, required=True,
                        choices=AGENT_PARTITIONS)
    parser.add_argument('--num', type=int, required=True,
                        choices=range(1, NUM_CHUNKS + 1))
    parser.add_argument('--values', action='store_true')
    args = parser.parse_args()
    chunk = args.num - 1

    # create output folder
    output_dir = SIM_DIR + '{}/{}/'.format(
        args.part, 'values' if args.values else 'outcomes')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # check if chunk has already been processed
    path = output_dir + '{}.pkl'.format(chunk)
    if os.path.isfile(path):
        print('Chunk {} already exists.'.format(chunk))
        exit(0)

    # generator
    gen_cls = ValueGenerator if args.values else OutcomeGenerator
    gen = gen_cls(env=SimulatorEnv)
    obj = gen.process_chunk(part=args.part, chunk=chunk)

    # save
    topickle(obj, path)


if __name__ == '__main__':
    main()
