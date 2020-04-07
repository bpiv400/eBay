import numpy as np
from compress_pickle import load, dump
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from train.best_models import extract_best_experiment
from processing.processing_consts import NUM_OUT
from constants import MODEL_DIR, LOG_DIR, INPUT_DIR, TEST, MODELS, PLOT_DIR


def get_baserate(y, num_out):
    p = np.array([(y == i).mean() for i in range(num_out)])
    p = p[p > 0]
    return np.sum(p * np.log(p))


def get_delay_baserate(y, num_out):
    counts = np.array([(y == i).sum() for i in range(num_out)],
                      dtype='float64')
    cens = np.array([(y == i).sum() for i in range(-num_out, 0)],
                    dtype='float64')
    for i in range(num_out):
        counts[i:] += cens[i] / (num_out - i)
    assert (np.abs(counts.sum() - len(y)) < 1e-8)
    p = counts / counts.sum()
    p_arrival = p[y[y >= 0]]
    p_cens = np.array([p[i:].sum() for i in y if i < 0])
    return np.log(np.concatenate([p_arrival, p_cens], axis=0)).mean()


def main():
    # loop over models, save training curves to dictionary
    lnL = dict()
    for m in MODELS:
        print(m)

        # initialize dictionary
        lnL[m] = dict()

        # number of periods
        num_out = NUM_OUT[m]
        if num_out == 1:
            num_out += 1

        # load data
        y = load(INPUT_DIR + '{}/{}.gz'.format(TEST, m))['y']

        if 'delay' in m or m == 'next_arrival':
            # initialization value
            p_arrival = np.ones((y >= 0).sum()) / num_out
            p_cens = -y[y < 0] / num_out
            lnL0 = np.log(np.concatenate([p_arrival, p_cens], axis=0)).mean()

            # baserate
            lnL[m]['baserate'] = get_delay_baserate(y, num_out)

        else:
            # initialization value
            lnL0 = np.log(1 / num_out)

            # baserate
            lnL[m]['baserate'] = get_baserate(y, num_out)

        # find best performing experiment
        em = EventMultiplexer().AddRunsFromDirectory(LOG_DIR + m)
        run = extract_best_experiment(em.Reload()) 
        for k in ['test', 'train']:
            lnL[m][k] = [lnL0] + [s.value for s in em.Scalars(run, 'lnL_' + k)]

    # save output
    dump(lnL, PLOT_DIR + 'lnL.pkl')


if __name__ == '__main__':
    main()
