import numpy as np
from compress_pickle import load, dump
from assess.assess_utils import get_num_out
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from train.best_models import extract_best_experiment
from constants import LOG_DIR, INPUT_DIR, TEST, MODELS, PLOT_DIR, CENSORED_MODELS


def get_baserate(y, num_out):
    p = np.array([(y == i).mean() for i in range(num_out)])
    p = p[p > 0]
    return np.sum(p * np.log(p))


def main():
    # loop over models, save training curves to dictionary
    lnl = dict()
    for m in MODELS:
        print(m)

        # initialize dictionary
        lnl[m] = dict()

        # number of periods
        num_out = get_num_out(m)

        # load data
        y = load(INPUT_DIR + '{}/{}.gz'.format(TEST, m))['y']

        if m in CENSORED_MODELS:
            # initialization value
            p_arrival = np.ones((y >= 0).sum()) / num_out
            p_cens = -y[y < 0] / num_out
            lnl0 = np.log(np.concatenate([p_arrival, p_cens], axis=0)).mean()

        else:
            # initialization value
            lnl0 = np.log(1 / num_out)

            # baserate
            lnl[m]['baserate'] = get_baserate(y, num_out)

        # find best performing experiment
        em = EventMultiplexer().AddRunsFromDirectory(LOG_DIR + m)
        run = extract_best_experiment(em.Reload()) 
        for k in ['test', 'train']:
            lnl[m][k] = [lnl0] + [s.value for s in em.Scalars(run, 'lnL_' + k)]

    # save output
    dump(lnl, PLOT_DIR + 'lnL.pkl')


if __name__ == '__main__':
    main()
