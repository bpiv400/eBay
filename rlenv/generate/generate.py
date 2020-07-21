import torch.multiprocessing as mp
import argparse
import torch
from constants import PARTS_DIR, TRAIN_RL, VALIDATION, TEST
from rlenv.generate.Generator import DiscrimGenerator
from rlenv.generate.util import process_sims
from utils import run_func_on_chunks, process_chunk_worker


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True,
                        choices=[TRAIN_RL, VALIDATION, TEST])
    part = parser.parse_args().part

    # process chunks in parallel
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(
            part=part,
            gen_class=DiscrimGenerator,
            gen_kwargs=dict(verbose=False)
        )
    )

    # concatenate, clean, and save
    process_sims(part=part, sims=sims, parent_dir=PARTS_DIR)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.set_default_dtype(torch.float32)
    main()
