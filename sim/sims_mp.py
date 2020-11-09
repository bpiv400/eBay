import argparse
import os
import pandas as pd
import torch
from rlenv.generate.Generator import OutcomeGenerator, ValueGenerator
from rlenv.generate.util import process_sims
from sim.envs import SimulatorEnv
from utils import run_func_on_chunks, process_chunk_worker, topickle
from constants import SIM_DIR
from featnames import AGENT_PARTITIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=AGENT_PARTITIONS)
    parser.add_argument('--values', action='store_true')
    args = parser.parse_args()

    # output folder
    output_dir = SIM_DIR + '{}/'.format(args.part)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # process chunks in parallel
    gen_cls = ValueGenerator if args.values else OutcomeGenerator
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(part=args.part,
                         gen_class=gen_cls,
                         gen_kwargs=dict(env=SimulatorEnv))
    )

    # concatenate, clean, and save
    if not args.values:
        process_sims(part=args.part, sims=sims, output_dir=output_dir)
    else:
        df = pd.concat(sims).sort_index()
        topickle(df, output_dir + 'values.pkl')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
