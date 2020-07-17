import torch.multiprocessing as mp
import argparse
import torch
from constants import PARTITIONS, AGENT_PARTS_DIR
from agent.eval.ValueGenerator import ValueGenerator
from rlenv.generate.Generator import DiscrimGenerator
from rlenv.generate.util import process_sims
from utils import run_func_on_chunks


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True, type=str)
    parser.add_argument('--values', action='store_true')
    parser.add_argument('--verbose', action='store_true',
                        help='print event detail')
    args = parser.parse_args()
    part, values, verbose = args.part, args.values, args.verbose
    assert part in PARTITIONS[1:]

    # create generator
    if values:
        cls = ValueGenerator
    else:
        cls = DiscrimGenerator
    gen = cls(verbose=verbose)

    # process chunks in parallel
    sims = run_func_on_chunks(
        f=gen.process_chunk,
        func_kwargs=dict(part=part)
    )

    # concatenate, clean, and save
    process_sims(part=part, sims=sims, parent_dir=AGENT_PARTS_DIR)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.set_default_dtype(torch.float32)
    main()
