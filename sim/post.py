import os
import h5py
import argparse
from shutil import copyfile
from compress_pickle import load, dump
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_multiplexer import \
    EventMultiplexer
from constants import MODEL_DIR, LOG_DIR, MODELS, DISCRIM_MODELS, \
    POLICY_MODELS, FIRST_ARRIVAL_MODEL, PARTS_DIR, TRAIN_RL, \
    VALIDATION, NO_ARRIVAL_CUTOFF, NUM_CHUNKS, NUM_RL_WORKERS
from utils import get_model_predictions, load_file
from featnames import START_PRICE, LOOKUP, X_LSTG, P_ARRIVAL, NO_ARRIVAL

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def get_cols(df):
    return [c.encode('utf-8') for c in list(df.columns)]


def save_agent(lookup=None, x_lstg=None, p_arrival=None):
    print('Saving {}/agent'.format(TRAIN_RL))

    # 32-bit
    lookup = lookup.reset_index(drop=False).astype('float32')

    # create directory
    out_dir = PARTS_DIR + '{}/agent/'.format(TRAIN_RL)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # split and save
    idx = np.arange(0, len(x_lstg), step=NUM_RL_WORKERS)
    for i in range(NUM_RL_WORKERS):
        f = h5py.File(out_dir + '{}.hdf5'.format(i), 'w')
        lookup_ds = f.create_dataset(LOOKUP,
                                     data=lookup.iloc[idx, :].values)
        x_lstg_ds = f.create_dataset(X_LSTG,
                                     data=x_lstg.iloc[idx, :].values)
        f.create_dataset(P_ARRIVAL, data=p_arrival.iloc[idx, :].values)
        lookup_ds.attrs['cols'] = get_cols(lookup)
        x_lstg_ds.attrs['cols'] = get_cols(x_lstg)
        f.close()

        # increment indices
        idx = idx + 1
        if idx[-1] >= len(x_lstg):
            idx = idx[:-1]


def save_chunks(lookup=None, x_lstg=None, p_arrival=None, part=None):
    print('Saving {}/chunks'.format(part))

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


def save_components(part):
    # load listing data
    x_lstg = load_file(part, X_LSTG)

    # predicted arrival probabilities
    p0 = get_model_predictions(FIRST_ARRIVAL_MODEL, x_lstg)
    x_lstg = pd.concat(x_lstg.values(), axis=1)

    # save no arrival probabilities
    s = pd.Series(p0[:, -1], index=x_lstg.index)
    dump(s, PARTS_DIR + '{}/{}.gz'.format(part, NO_ARRIVAL))

    # listings to keep
    keep = p0[:, -1] < NO_ARRIVAL_CUTOFF

    # data frame of arrival probabilities
    p_arrival = pd.DataFrame(p0, index=x_lstg.index)

    # load lookup, drop infrequent arrivals, and sort by start_price
    lookup = load_file(part, LOOKUP)[keep].sort_values(by=START_PRICE)

    # reindex other dataframes
    x_lstg = x_lstg.reindex(index=lookup.index).astype('float32')
    assert x_lstg.isna().sum().sum() == 0
    assert (x_lstg.index == lookup.index).all()
    p_arrival = p_arrival.reindex(index=lookup.index).astype('float32')
    assert (abs(p_arrival.sum(axis=1) - 1.) < 1e8).all()
    assert (p_arrival.index == lookup.index).all()

    # chunks
    save_chunks(lookup=lookup,
                x_lstg=x_lstg,
                p_arrival=p_arrival,
                part=part)

    # agent training
    if part == TRAIN_RL:
        save_agent(lookup=lookup, x_lstg=x_lstg, p_arrival=p_arrival)


def extract_best_run(m):
    # load tensorboard log
    em = EventMultiplexer().AddRunsFromDirectory(LOG_DIR + m).Reload()
    # list of final log-likelihoods in holdout
    lnl = []
    for run, d in em.Runs().items():
        lnl.append(em.Scalars(run, 'lnL_test')[-1].value)
    # find best run
    idx = np.argmax(lnl)
    run = list(em.Runs().keys())[idx]
    return run


def main():
    # command line parameter for model group
    parser = argparse.ArgumentParser()
    parser.add_argument('--discrim', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    args = parser.parse_args()

    # model group
    if args.discrim:
        models = DISCRIM_MODELS
    else:
        models = MODELS + POLICY_MODELS

    # create dropout file
    dropout_path = MODEL_DIR + 'dropout.pkl'
    if not os.path.isfile(dropout_path):
        s = pd.Series(name='dropout')
    else:
        s = load(dropout_path)

    # loop over models
    for m in models:
        if not args.nosave:
            run = extract_best_run(m)  # best performing run
            print('{}: {}'.format(m, run))

            # copy best performing model into parent directory
            copyfile(MODEL_DIR + '{}/{}.net'.format(m, run),
                     MODEL_DIR + '{}.net'.format(m))

            # save dropout
            elem = run.split('_')
            dropout = (float(elem[-2]) / 10, float(elem[-1]) / 10)
            s.loc[m] = dropout
            dump(s, dropout_path)  # save dropout file

    # save chunks
    if not args.discrim:
        for part in [TRAIN_RL, VALIDATION]:
         save_components(part)




if __name__ == '__main__':
    main()
