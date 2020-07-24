import os
import argparse
from shutil import copyfile
from compress_pickle import load, dump
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_multiplexer import \
    EventMultiplexer
from constants import MODEL_DIR, LOG_DIR, MODELS, DISCRIM_MODEL, \
    POLICY_MODELS, DROPOUT


def get_lnl(em=None, run=None, name=None):
    tuples = em.Scalars(run, 'lnL_{}'.format(name))
    lnl = [tuples[i].value for i in range(len(tuples))]
    return lnl


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
    # log-likelihoods
    lnl_test = get_lnl(em=em, run=run, name='test')
    lnl_train = get_lnl(em=em, run=run, name='train')
    return run, lnl_test, lnl_train


def main():
    # command line parameter for model group
    parser = argparse.ArgumentParser()
    parser.add_argument('--discrim', action='store_true')
    args = parser.parse_args()

    # model group
    if args.discrim:
        models = [DISCRIM_MODEL]
    else:
        models = MODELS + POLICY_MODELS

    # create dropout file
    dropout_path = MODEL_DIR + 'dropout.pkl'
    if not os.path.isfile(dropout_path):
        s = pd.Series(name=DROPOUT)
    else:
        s = load(dropout_path)

    # loop over models
    for m in models:
        run, _, _ = extract_best_run(m)  # best performing run
        print('{}: {}'.format(m, run))

        # copy best performing model into parent directory
        copyfile(MODEL_DIR + '{}/{}.net'.format(m, run),
                 MODEL_DIR + '{}.net'.format(m))

        # save dropout
        elem = run.split('_')
        dropout = (float(elem[-2]) / 10, float(elem[-1]) / 10)
        s.loc[m] = dropout
        dump(s, dropout_path)  # save dropout file


if __name__ == '__main__':
    main()
