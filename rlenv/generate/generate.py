from multiprocessing import Pool
import psutil
import argparse
import torch
import time
from constants import PARTITIONS, NUM_CHUNKS
from agent.eval.ValueGenerator import ValueGenerator
from rlenv.generate.Generator import DiscrimGenerator
from rlenv.generate.util import process_sims


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True, type=str)
    parser.add_argument('--values', action='store_true')
    parser.add_argument('--verbose', action='store_true',
                        help='print event detail')
    args = parser.parse_args()
    part, values, verbose = args.part, args.values, args.verbose
    assert part in PARTITIONS

    # create generator
    if values:
        cls = ValueGenerator
    else:
        cls = DiscrimGenerator
    gen = cls(part=part, verbose=verbose)

    # process chunks in parallel
    num_workers = min(NUM_CHUNKS, psutil.cpu_count() - 1)
    pool = Pool(num_workers)
    jobs = [pool.apply_async(gen.process_chunk, (part, i))
            for i in range(NUM_CHUNKS)]
    sims = []
    for job in jobs:
        while True:
            if job.ready():
                sims.append(job.get())
                continue
            else:
                time.sleep(5)

    # concatenate, clean, and save
    process_sims(part=part, sims=sims)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    main()
