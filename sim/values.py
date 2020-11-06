import os
import torch
import pandas as pd
from rlenv.generate.Generator import ValueGenerator
from sim.envs import SimulatorEnv
from utils import run_func_on_chunks, process_chunk_worker, topickle, input_partition
from constants import SIM_DIR


def main():
    part = input_partition(agent=True)

    # create output folder
    output_dir = SIM_DIR + '{}/'.format(part)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # process chunks in parallel
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(part=part,
                         gen_class=ValueGenerator,
                         gen_kwargs=dict(env=SimulatorEnv))
    )

    # concat and save output
    df = pd.concat(sims, axis=0).sort_index()
    topickle(df, output_dir + 'values.pkl')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
