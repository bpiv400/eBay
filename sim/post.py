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
from featnames import START_PRICE, LOOKUP, X_LSTG

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def get_cols(df):
    return [c.encode('utf-8') for c in list(df.columns)]


def save_chunks(lookup=None, x_lstg=None, chunk_dir=None, agent=None):
    # 32-bit and number of files
    if agent:
        lookup = lookup.astype('float32')
        x_lstg = x_lstg.astype('float32')
        num_files = NUM_RL_WORKERS * 2
    else:
        num_files = NUM_CHUNKS

    # split and save
    idx = np.arange(0, len(x_lstg), step=num_files)
    for i in range(num_files):
        if agent:
            # save to file
            f = h5py.File(chunk_dir + '{}.hdf5'.format(i), 'w')
            lookup_ds = f.create_dataset(
                LOOKUP, data=lookup.iloc[idx, :].values)
            x_lstg_ds = f.create_dataset(
                X_LSTG, data=x_lstg.iloc[idx, :].values)
            lookup_ds.attrs['cols'] = get_cols(lookup)
            x_lstg_ds.attrs['cols'] = get_cols(x_lstg)
            f.close()
        else:
            chunk = {LOOKUP: lookup.iloc[idx, :],
                     X_LSTG: x_lstg.iloc[idx, :]}
            path = chunk_dir + '{}.gz'.format(i + 1)
            dump(chunk, path)

        # increment indices
        idx = idx + 1
        if idx[-1] >= len(x_lstg):
            idx = idx[:-1]


def create_chunks(part):
    # load listing data
    x_lstg = load_file(part, X_LSTG)

    # model predictions
    p0 = get_model_predictions(FIRST_ARRIVAL_MODEL, x_lstg)
    keep = p0 < NO_ARRIVAL_CUTOFF

    # concatenate x into one dataframe
    x_lstg = pd.concat(x_lstg.values(), axis=1)

    # load lookup, drop infrequent arrivals, and sort by start_price
    lookup = load_file(part, LOOKUP)[keep].sort_values(by=START_PRICE)
    x_lstg = x_lstg.reindex(index=lookup.index)
    lookup = lookup.reset_index(drop=False)
    assert x_lstg.isna().sum().sum() == 0

    # make chunk directory
    chunk_dir = PARTS_DIR + '{}/chunks/'.format(part)
    if not os.path.isdir(chunk_dir):
        os.mkdir(chunk_dir)

    # save chunks
    save_chunks(lookup=lookup, x_lstg=x_lstg,
                chunk_dir=chunk_dir, agent=False)
    save_chunks(lookup=lookup, x_lstg=x_lstg,
                chunk_dir=chunk_dir, agent=True)


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
            create_chunks(part)


if __name__ == '__main__':
    main()
