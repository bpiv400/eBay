from shutil import copyfile
import numpy as np
from tensorboard.backend.event_processing.event_multiplexer import \
    EventMultiplexer
from constants import MODEL_DIR, LOG_DIR, MODELS, DISCRIM_MODELS


def extract_best_experiment(em):
    # list of final log-likelihoods
    lnl = []
    for run, d in em.Runs().items():
        lnl.append(em.Scalars(run, 'lnL_test')[-1].value)
    # find index of best
    idx = np.argmax(lnl)
    return list(em.Runs().keys())[idx]


def main():
    # for each model, choose best experiment
    for m in MODELS + DISCRIM_MODELS + ['init_slr']:
        em = EventMultiplexer().AddRunsFromDirectory(LOG_DIR + m).Reload()

        if len(em.Runs().keys()) == 0:
            continue

        # find best performing experiment
        run = extract_best_experiment(em)

        # copy best performing model into parent directory
        print('{}: {}'.format(m, run))
        copyfile(MODEL_DIR + '{}/{}.net'.format(m, run),
                 MODEL_DIR + '{}.net'.format(m))


if __name__ == '__main__':
    main()
