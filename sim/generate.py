import argparse
import torch
from rlenv.generate.Generator import OutcomeGenerator
from rlenv.generate.util import process_sims
from sim.envs import SimulatorEnv
from utils import run_func_on_chunks, process_chunk_worker
from constants import SIM_DIR
from featnames import AGENT_PARTITIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=AGENT_PARTITIONS)
    args = parser.parse_args()

    # environment and output folder
    env = SimulatorEnv
    output_dir = SIM_DIR + '{}/'.format(args.part)

    # process chunks in parallel
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(part=args.part,
                         gen_class=OutcomeGenerator,
                         gen_kwargs=dict(env=env))
    )

    # concatenate, clean, and save
    process_sims(part=args.part, sims=sims, output_dir=output_dir)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
