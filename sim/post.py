import os
import argparse
from shutil import copyfile
from compress_pickle import load, dump
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_multiplexer import \
    EventMultiplexer
from processing.util import get_lstgs
from processing.c_frames.lookup import create_lookup
from processing.c_frames.lstg import create_x_lstg
from processing.c_frames.offer import create_x_offer
from processing.c_frames.thread import create_x_thread
from constants import MODEL_DIR, LOG_DIR, MODELS, DISCRIM_MODELS, \
    POLICY_MODELS, FIRST_ARRIVAL_MODEL, AGENT_PARTS_DIR, TRAIN_RL, \
    VALIDATION, NO_ARRIVAL_CUTOFF, NUM_CHUNKS, DROPOUT
from utils import get_model_predictions
from featnames import START_PRICE, LOOKUP, X_LSTG, P_ARRIVAL


def save_chunks(lookup=None, x_lstg=None, p_arrival=None, part=None):
    print('Saving {}/chunks/'.format(part))

    # put x_lstg in single dataframe
    x_lstg = pd.concat(x_lstg.values(), axis=1)
    assert x_lstg.isna().sum().sum() == 0

    # sort by start_price
    lookup = lookup.sort_values(by=START_PRICE)
    x_lstg = x_lstg.reindex(index=lookup.index)
    p_arrival = p_arrival.reindex(index=lookup.index)

    # create directory
    out_dir = AGENT_PARTS_DIR + '{}/chunks/'.format(part)
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
    # lookup and x_lstg
    lstgs = get_lstgs(part)
    lookup = create_lookup(lstgs=lstgs)
    x_lstg = create_x_lstg(lstgs=lstgs)

    # predicted arrival probabilities
    print('Predicting arrival distributions for {} partition'.format(part))
    p_arrival = get_model_predictions(FIRST_ARRIVAL_MODEL, x_lstg)
    keep = p_arrival[:, -1] < NO_ARRIVAL_CUTOFF
    p_arrival = pd.DataFrame(p_arrival,
                             index=lookup.index,
                             dtype='float32')
    assert (abs(p_arrival.sum(axis=1) - 1.) < 1e8).all()

    # lookup
    print('Saving {}/'.format(part))
    lookup = lookup[keep]
    dump(lookup, AGENT_PARTS_DIR + '{}/{}.gz'.format(part, LOOKUP))

    # x_lstg
    x_lstg = {k: v.reindex(index=lookup.index) for k, v in x_lstg.items()}
    to_save = {k: v.values for k, v in x_lstg.items()}
    dump(to_save, AGENT_PARTS_DIR + '{}/{}.pkl'.format(part, X_LSTG))

    # x_offer and clock
    x_offer, clock = create_x_offer(lstgs=lookup.index)
    dump(x_offer, AGENT_PARTS_DIR + '{}/x_offer.gz'.format(part))
    dump(clock, AGENT_PARTS_DIR + '{}/clock.gz'.format(part))

    # x_thread
    x_thread = create_x_thread(lstgs=lookup.index)
    dump(x_thread, AGENT_PARTS_DIR + '{}/x_thread.gz'.format(part))

    # first arrival distribution
    p_arrival = p_arrival.reindex(index=lookup.index)

    # chunks
    save_chunks(lookup=lookup,
                x_lstg=x_lstg,
                p_arrival=p_arrival,
                part=part)


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
    parser.add_argument('--rl', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    args = parser.parse_args()

    # model group
    if args.rl:
        models = DISCRIM_MODELS + POLICY_MODELS
    else:
        models = MODELS

    # create dropout file
    dropout_path = MODEL_DIR + 'dropout.pkl'
    if not os.path.isfile(dropout_path):
        s = pd.Series(name=DROPOUT)
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
    if not args.rl:
        for part in [TRAIN_RL, VALIDATION]:
            save_components(part)


if __name__ == '__main__':
    main()
