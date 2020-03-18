from compress_pickle import load, dump
import numpy as np
from processing.processing_consts import NUM_OUT
from constants import INPUT_DIR, VALIDATION, MODELS, PLOT_DATA_DIR


def get_baserate(y, periods):
    p = np.array([(y == i).mean() for i in range(periods)])
    p = p[p > 0]
    return np.sum(p * np.log(p))


def get_delay_baserate(y, periods):
    counts = np.array([(y == i).sum() for i in range(periods)],
                      dtype='float64')
    cens = np.array([(y == i).sum() for i in range(-periods, 0)],
                    dtype='float64')
    for i in range(periods):
        counts[i:] += cens[i] / (periods - i)
    assert (np.abs(counts.sum() - len(y)) < 1e-8)
    p = counts / counts.sum()
    p_arrival = p[y[y >= 0]]
    p_cens = np.array([p[i:].sum() for i in y if i < 0])
    return np.log(np.concatenate([p_arrival, p_cens], axis=0)).mean()


def main():
    # initialize output dataframe
    lnL0, lnL_bar = dict(), dict()

    # calculate initialization value and baserate for each model
    for m in MODELS:
        print(m)

        # number of periods
        periods = NUM_OUT[m]
        if periods == 1:
            periods += 1

        # load data
        y = load(INPUT_DIR + '{}/{}.gz'.format(VALIDATION, m))['y']

        if 'delay' in m or m == 'next_arrival':
            # initialization value
            p_arrival = np.ones((y >= 0).sum()) / periods
            p_cens = -y[y < 0] / periods
            lnL0[m] = np.log(np.concatenate([p_arrival, p_cens], axis=0)).mean()

            # baserate
            lnL_bar[m] = get_delay_baserate(y, periods)

        else:
            # initialization value
            lnL0[m] = np.log(1 / periods)

            # baserate
            lnL_bar[m] = get_baserate(y, periods)

    # save output
    dump(lnL0, PLOT_DATA_DIR + '{}/{}.pkl'.format('lnL', 'lnL0'))
    dump(lnL_bar, PLOT_DATA_DIR + '{}/{}.pkl'.format('lnL', 'lnL_bar'))


if __name__ == '__main__':
    main()
