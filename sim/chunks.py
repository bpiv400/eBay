import os
import pandas as pd
from sim.EBayDataset import EBayDataset
from utils import topickle, get_model_predictions, load_file, load_featnames, \
    input_partition
from constants import FIRST_ARRIVAL_MODEL, PARTS_DIR, NUM_CHUNKS, INTERVAL_CT_ARRIVAL
from featnames import LOOKUP, X_LSTG, P_ARRIVAL, META, START_PRICE, START_TIME, \
    DEC_PRICE


def save_chunks(p_arrival=None, part=None, lookup=None):
    # x_lstg
    x_lstg = load_file(part, X_LSTG)
    featnames = load_featnames(X_LSTG)
    x_lstg = {k: pd.DataFrame(v, columns=featnames[k], index=lookup.index)
              for k, v in x_lstg.items()}
    x_lstg = pd.concat(x_lstg.values(), axis=1)  # single DataFrame
    assert x_lstg.isna().sum().sum() == 0

    # drop extraneous lookup columns
    lookup = lookup[[META, START_PRICE, DEC_PRICE, START_TIME]]

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
    for i in range(NUM_CHUNKS):
        idx_i = [idx[k] for k in range(len(idx))
                 if k % NUM_CHUNKS == i]
        chunk = {LOOKUP: lookup.reindex(index=idx_i),
                 X_LSTG: x_lstg.reindex(index=idx_i),
                 P_ARRIVAL: p_arrival.reindex(index=idx_i)}
        topickle(chunk, out_dir + '{}.pkl'.format(i))


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
    part = input_partition()
    print('Saving {} chunks'.format(part))

    # lookup file
    lookup = load_file(part, LOOKUP)

    # arrival probabilities
    p_arrival = get_p_arrival(part=part, lookup=lookup)

    # save chunks
    save_chunks(p_arrival=p_arrival, part=part, lookup=lookup)


if __name__ == '__main__':
    main()
