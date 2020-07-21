import os
import torch.multiprocessing as mp
from compress_pickle import dump
import numpy as np
import pandas as pd
from rlenv.generate.Generator import DiscrimGenerator
from rlenv.generate.util import process_sims
from utils import get_model_predictions, load_file, load_featnames, \
    run_func_on_chunks, process_chunk_worker, input_partition
from constants import FIRST_ARRIVAL_MODEL, PARTS_DIR, NUM_CHUNKS
from featnames import START_PRICE, LOOKUP, X_LSTG, P_ARRIVAL, END_TIME


def save_chunks(lookup=None, x_lstg=None, p_arrival=None, part=None):
    # put x_lstg in single dataframe
    x_lstg = pd.concat(x_lstg.values(), axis=1)
    assert x_lstg.isna().sum().sum() == 0

    # sort by start_price
    lookup = lookup.drop(END_TIME, axis=1).sort_values(by=START_PRICE)
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


def create_chunks(part):
    # lookup and x_lstg
    lookup = load_file(part, LOOKUP)
    x_lstg = load_file(part, X_LSTG)
    featnames = load_featnames(X_LSTG)
    x_lstg = {k: pd.DataFrame(v, columns=featnames[k], index=lookup.index)
              for k, v in x_lstg.items()}

    # predicted arrival probabilities
    p_arrival = get_model_predictions(FIRST_ARRIVAL_MODEL, x_lstg)
    p_arrival = pd.DataFrame(p_arrival,
                             index=lookup.index,
                             dtype='float32')
    assert (abs(p_arrival.sum(axis=1) - 1.) < 1e8).all()

    # chunks
    save_chunks(lookup=lookup,
                x_lstg=x_lstg,
                p_arrival=p_arrival,
                part=part)


def main():
    # command line parameter for model group
    part = input_partition()

    # save chunks
    print('Saving {} chunks'.format(part))
    create_chunks(part)

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


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
