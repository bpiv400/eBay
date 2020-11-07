import argparse
import os
from rlenv.generate.Generator import ValueGenerator
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
    args = parser.parse_args()
    chunk = args.num - 1

    # create output folder
    output_dir = SIM_DIR + '{}/values/'.format(args.part)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # process one chunk
    path = output_dir + '{}.pkl'.format(chunk)
    if os.path.isfile():
        print('Values for chunk {} already exists.'.format(chunk))
        exit(0)
    gen = ValueGenerator(env=SimulatorEnv)
    df = gen.process_chunk(part=args.part, chunk=chunk)
    topickle(df, path)


if __name__ == '__main__':
    main()
