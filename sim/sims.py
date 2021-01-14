import argparse
import os
import pandas as pd
from rlenv.generate.Generator import OutcomeGenerator, ValueGenerator, SimulatorEnv
from rlenv.generate.util import process_sims
from utils import run_func_on_chunks, process_chunk_worker, topickle
from constants import SIM_DIR, NUM_CHUNKS
from featnames import AGENT_PARTITIONS, TEST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--num', type=int, choices=range(1, NUM_CHUNKS + 1))
    parser.add_argument('--values', action='store_true')
    args = parser.parse_args()

    # generator
    gen_cls = ValueGenerator if args.values else OutcomeGenerator

    if args.num is not None:
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
        gen = gen_cls(env=SimulatorEnv)
        obj = gen.process_chunk(part=args.part, chunk=chunk)

        # save
        topickle(obj, path)

    else:
        sims = run_func_on_chunks(
            f=process_chunk_worker,
            func_kwargs=dict(part=args.part,
                             gen_class=gen_cls,
                             gen_kwargs=dict(env=SimulatorEnv))
        )

        # concatenate, clean, and save
        output_dir = SIM_DIR + '{}/'.format(args.part)
        if not args.values:
            process_sims(part=args.part, sims=sims, output_dir=output_dir)
        else:
            df = pd.concat(sims).sort_index()
            topickle(df, output_dir + 'values.pkl')


if __name__ == '__main__':
    main()
