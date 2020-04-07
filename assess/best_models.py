from shutil import copyfile
import numpy as np
from compress_pickle import dump
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from constants import MODEL_DIR, LOG_DIR, MODELS, DISCRIM_MODELS, PLOT_DIR


def extract_best_experiment(em):
    # initialize output dictionary, best marker, and name of run to keep
    best, lnL, keep = -np.inf, dict(), ''
    # loop over experiments, save best
    for run, d in em.Runs().items():
        lnL_test1 = em.Scalars(run, 'lnL_test')[-1].value
        if len(d['scalars']) > 0:
            curr = lnL_test1
            if curr > best:
                best = curr
                keep = run
                for k in ['lnL_test', 'lnL_train']:
                    lnL[k.split('_')[-1]] = [s.value for s in em.Scalars(run, k)]
    return lnL, keep


def main():
    # for each model, choose best experiment
    lnL = dict()
    for m in MODELS + DISCRIM_MODELS + ['init_slr']:
        em = EventMultiplexer().AddRunsFromDirectory(LOG_DIR + m)

        # find best performing experiment
        lnL[m], run = extract_best_experiment(em.Reload())      

        # copy best performing model into parent directory
        print('{}: {}'.format(m, run))
        copyfile(MODEL_DIR + '{}/{}.net'.format(m, run),
                 MODEL_DIR + '{}.net'.format(m))

    # save output
    dump(lnL, PLOT_DIR + 'lnL.pkl')


if __name__ == '__main__':
    main()
