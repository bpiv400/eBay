import argparse
from shutil import rmtree
import pandas as pd
from rlenv.generate.util import process_sims
from utils import unpickle, topickle
from constants import SIM_DIR, NUM_CHUNKS
from featnames import AGENT_PARTITIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=AGENT_PARTITIONS)
    parser.add_argument('--type', type=str, choices=['values', 'outcomes'])
    args = parser.parse_args()

    output_dir = SIM_DIR + '{}/'.format(args.part)
    sub_dir = output_dir + '{}/'.format(args.type)

    # concatenate
    sims = []
    for i in range(NUM_CHUNKS):
        chunk_path = sub_dir + '{}.pkl'.format(i)
        sims.append(unpickle(chunk_path))

    # concatenate, clean, and save
    if args.type == 'outcomes':
        process_sims(part=args.part, sims=sims, sim_dir=output_dir)
    else:
        df = pd.concat(sims).sort_index()
        topickle(df, output_dir + 'values.pkl')

    # delete individual files
    rmtree(sub_dir)


if __name__ == '__main__':
    main()
