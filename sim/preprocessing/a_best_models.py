import argparse
from shutil import copyfile
import numpy as np
from tensorboard.backend.event_processing.event_multiplexer import \
    EventMultiplexer
from constants import MODEL_DIR, LOG_DIR, MODELS, INIT_POLICY_MODELS, \
    INIT_VALUE_MODELS, DISCRIM_MODELS


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
    if post:
        models = INIT_VALUE_MODELS + DISCRIM_MODELS
    else:
        models = MODELS + INIT_POLICY_MODELS

    # for each model, choose best experiment
    for m in models:
        extract_best_run(m)


if __name__ == '__main__':
    main()
