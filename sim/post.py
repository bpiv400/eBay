import argparse
import os
from compress_pickle import dump
import numpy as np
import pandas as pd
from rlenv.generate.Generator import DiscrimGenerator
from rlenv.generate.util import process_sims
from sim.EBayDataset import EBayDataset
from utils import get_model_predictions, load_file, load_featnames, \
    run_func_on_chunks, process_chunk_worker, set_gpu_workers
from constants import FIRST_ARRIVAL_MODEL, PARTS_DIR, NUM_CHUNKS, \
    TRAIN_RL, VALIDATION, TEST
from featnames import START_PRICE, LOOKUP, X_LSTG, P_ARRIVAL, END_TIME


def simulate(part=None):
    # process chunks in parallel
    print('Generating {} discriminator input'.format(part))
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


def save_chunks(p_arrival=None, part=None, lookup=None):
    # x_lstg
    x_lstg = load_file(part, X_LSTG)
    featnames = load_featnames(X_LSTG)
    x_lstg = {k: pd.DataFrame(v, columns=featnames[k], index=lookup.index)
              for k, v in x_lstg.items()}
    x_lstg = pd.concat(x_lstg.values(), axis=1)  # single DataFrame
    assert x_lstg.isna().sum().sum() == 0

    # sort by start_price
    lookup = lookup.sort_values(by=START_PRICE)
    x_lstg = x_lstg.reindex(index=lookup.index)
    p_arrival = p_arrival.reindex(index=lookup.index)

    # create directory
    out_dir = PARTS_DIR + '{}/chunks/'.format(part)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # split and save
    idx = np.arange(0, len(x_lstg), step=NUM_CHUNKS)
    for i in range(NUM_CHUNKS):
        chunk = {LOOKUP: lookup.iloc[idx, :],
                 X_LSTG: x_lstg.iloc[idx, :],
                 P_ARRIVAL: p_arrival.iloc[idx, :]}
        path = out_dir + '{}.gz'.format(i)
        dump(chunk, path)

        # increment indices
        idx = idx + 1
        if idx[-1] >= len(x_lstg):
            idx = idx[:-1]


def get_p_arrival(part=None, lookup=None):
    data = EBayDataset(part=part, name=FIRST_ARRIVAL_MODEL)
    p_arrival = get_model_predictions(data)
    p_arrival = pd.DataFrame(p_arrival,
                             index=lookup.index,
                             dtype='float32')
    assert (abs(p_arrival.sum(axis=1) - 1.) < 1e8).all()
    return p_arrival


def main():
    # command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=[TRAIN_RL, VALIDATION, TEST])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    part, gpu = args.part, args.gpu

    # set gpu
    set_gpu_workers(gpu=gpu, spawn=True)

    # lookup file
    lookup = load_file(part, LOOKUP).drop(END_TIME, axis=1)

    # arrival probabilities
    p_arrival = get_p_arrival(part=part, lookup=lookup)

    # save chunks
    print('Saving {} chunks'.format(part))
    save_chunks(p_arrival=p_arrival, part=part, lookup=lookup)

    # generate discrim input
    simulate(part=part)


if __name__ == '__main__':
    main()
