import numpy as np
from compress_pickle import load, dump
from sim.best_models import extract_best_run
from inputs.const import NUM_OUT
from constants import INPUT_DIR, TEST, MODELS, PLOT_DIR, CENSORED_MODELS


def get_num_out(m):
    return NUM_OUT[m] if NUM_OUT[m] > 1 else 2


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

            # baserate
            lnl[m]['baserate'] = get_delay_baserate(y, num_out)

        else:
            # initialization value
            lnl0 = np.log(1 / num_out)

            # baserate
            lnl[m]['baserate'] = get_baserate(y, num_out)

        # find best performing experiment
        run, lnl_test, lnl_train = extract_best_run(m)
        lnl[m]['test'] = [lnl0] + lnl_test
        lnl[m]['train'] = [lnl0] + lnl_train

    # save output
    dump(lnl, PLOT_DIR + 'training_curves.pkl')


if __name__ == '__main__':
    main()
