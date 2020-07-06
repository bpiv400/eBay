import os
import argparse
from shutil import copyfile
from compress_pickle import load, dump
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_multiplexer import \
    EventMultiplexer
from constants import MODEL_DIR, LOG_DIR, MODELS, DISCRIM_MODELS, \
    POLICY_MODELS, FIRST_ARRIVAL_MODEL, PARTS_DIR, TRAIN_RL, VALIDATION
from utils import get_model_predictions, load_file
from featnames import NO_ARRIVAL


def save_no_arrival_prob(part):
    # load data as dictionary of 32-bit numpy arrays
    x = load_file(part, 'x_lstg')
    idx = x['lstg'].index
    x = {k: v.astype('float32').values for k, v in x.items()}

    # model predictions
    p = get_model_predictions(FIRST_ARRIVAL_MODEL, x)

    # create series
    s = pd.Series(p[:, -1], index=idx, name=NO_ARRIVAL)

    # save
    dump(s, PARTS_DIR + '{}/p_no_arrival.gz'.format(part))


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

    # create series of no-arrival probability
    if not args.discrim:
        for part in [TRAIN_RL, VALIDATION]:
            save_no_arrival_prob(part)


if __name__ == '__main__':
    main()
