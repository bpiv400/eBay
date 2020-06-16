import argparse
from shutil import copyfile
from compress_pickle import dump
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_multiplexer import \
    EventMultiplexer
from constants import MODEL_DIR, LOG_DIR, MODELS, DISCRIM_MODELS, \
    INIT_MODELS, FIRST_ARRIVAL_MODEL, PARTS_DIR, TRAIN_RL, VALIDATION
from utils import get_model_predictions, load_file


def save_no_arrival_prob(part):
    # load data as dictionary of 32-bit numpy arrays
    x = load_file(part, 'x_lstg')
    idx = x['lstg'].index
    x = {k: v.astype('float32').values for k, v in x.items()}

    # model predictions
    p = get_model_predictions(FIRST_ARRIVAL_MODEL, x)

    # create series
    s = pd.Series(p[:, -1], index=idx, name='p_no_arrival')

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
    # copy best performing model into parent directory
    print('{}: {}'.format(m, run))
    copyfile(MODEL_DIR + '{}/{}.net'.format(m, run),
             MODEL_DIR + '{}.net'.format(m))


def main():
    # command line parameter for model group
    parser = argparse.ArgumentParser()
    parser.add_argument('--post', action='store_true')
    post = parser.parse_args().post

    # model group
    models = INIT_MODELS + DISCRIM_MODELS if post else MODELS

    # for each model, choose best experiment
    for m in models:
        extract_best_run(m)

    # create series of no-arrival probability
    if not post:
        for part in [TRAIN_RL, VALIDATION]:
            save_no_arrival_prob(part)


if __name__ == '__main__':
    main()
