import argparse
from shutil import copyfile
import numpy as np
from compress_pickle import dump
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from train.train_consts import LNLR1
from constants import MODEL_DIR,LOG_DIR, MODELS, DISCRIM_MODELS, PLOT_DATA_DIR


def extract_best_experiment(em):
    # initialize output dictionary, best marker, and name of run to keep
    best, lnL, keep = -np.inf, dict(), ''
    # loop over experiments, save best
    for run, d in em.Runs().items():
        lnlr1 = em.Scalars(run, 'lnlr')[-1].value
        lnL_test1 = em.Scalars(run, 'lnL_test')[-1].value
        if len(d['scalars']) > 0 and lnlr1 == LNLR1:
            curr = lnL_test1
            if curr > best:
                best = curr
                keep = run
                for k in ['lnL_test', 'lnL_train']:
                    lnL[k.split('_')[-1]] = [s.value for s in em.Scalars(run, k)]
    return lnL, keep


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='models')
    args = parser.parse_args()

    # translate to set
    if args.set == 'models':
        group = MODELS
    elif args.set == 'discrim':
        group = DISCRIM_MODELS
    elif args.set == 'init':
        group = ['init_slr']
    else:
        raise RuntimeError('Invalid set: {}'.format(args.set))

    # for each model, choose best experiment
    lnL = dict()
    for m in group:
        em = EventMultiplexer().AddRunsFromDirectory(LOG_DIR + m)

        # find best performing experiment
        lnL[m], run = extract_best_experiment(em.Reload())      

        # copy best performing model into parent directory
        print('{}: {}'.format(m, run))
        copyfile(MODEL_DIR + '{}/{}.net'.format(m, run),
                 MODEL_DIR + '{}.net'.format(m))

    # save output
    dump(lnL, PLOT_DATA_DIR + 'lnL.pkl')


if __name__ == '__main__':
    main()
