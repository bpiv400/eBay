import argparse
import os
from compress_pickle import dump
import pandas as pd
from sim.EBayDataset import EBayDataset
from utils import get_model_predictions, load_file, load_featnames, \
    set_gpu_workers
from constants import FIRST_ARRIVAL_MODEL, PARTS_DIR, NUM_RL_WORKERS, \
    TRAIN_RL, VALIDATION, TEST, INTERVAL_CT_ARRIVAL
from featnames import SLR_BO_CT, LOOKUP, X_LSTG, P_ARRIVAL, END_TIME


def save_chunks(p_arrival=None, part=None, lookup=None):
    print('Saving {} chunks'.format(part))

    # x_lstg
    x_lstg = load_file(part, X_LSTG)
    featnames = load_featnames(X_LSTG)
    x_lstg = {k: pd.DataFrame(v, columns=featnames[k], index=lookup.index)
              for k, v in x_lstg.items()}
    x_lstg = pd.concat(x_lstg.values(), axis=1)  # single DataFrame
    assert x_lstg.isna().sum().sum() == 0

    # drop extraneous lookup columns
    lookup.drop([END_TIME, SLR_BO_CT], axis=1, inplace=True)

    # sort by no arrival probability
    p_arrival = p_arrival.sort_values(INTERVAL_CT_ARRIVAL)
    x_lstg = x_lstg.reindex(index=p_arrival.index)
    lookup = lookup.reindex(index=p_arrival.index)

    # create directory
    out_dir = PARTS_DIR + '{}/chunks/'.format(part)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # split into chunks
    idx = lookup.index
    for i in range(NUM_RL_WORKERS):
        idx_i = [idx[k] for k in range(len(idx))
                 if k % NUM_RL_WORKERS == i]
        chunk = {LOOKUP: lookup.reindex(index=idx_i),
                 X_LSTG: x_lstg.reindex(index=idx_i),
                 P_ARRIVAL: p_arrival.reindex(index=idx_i)}
        dump(chunk, out_dir + '{}.gz'.format(i))


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
    lookup = load_file(part, LOOKUP)

    # arrival probabilities
    p_arrival = get_p_arrival(part=part, lookup=lookup)

    # save chunks
    save_chunks(p_arrival=p_arrival, part=part, lookup=lookup)


if __name__ == '__main__':
    main()
