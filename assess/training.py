import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import loggamma, expit
from sim.best_models import extract_best_run
from sim.EBayDataset import EBayDataset
from utils import load_inputs, topickle, get_model_predictions
from inputs.const import NUM_OUT
from constants import PLOT_DIR
from featnames import TEST, MODELS, CENSORED_MODELS, DISCRIM_MODEL, BYR_HIST_MODEL


def get_auc(s):
    fp = s.index.values
    fp_delta = fp[1:] - fp[:-1]
    tp = s.values
    tp_bar = (tp[1:] + tp[:-1]) / 2
    auc = (fp_delta * tp_bar).sum()
    return auc


def get_roc():
    # vectors of predicted probabilities, by ground truth
    data = EBayDataset(part=TEST, name=DISCRIM_MODEL)
    y = data.d['y']
    p = get_model_predictions(data)
    p = p[:, 1]
    p0, p1 = p[y == 0], p[y == 1]

    # sweep out ROC curve
    s = pd.Series()
    dim = np.arange(0, 1 + 1e-8, 0.001)
    for tau in dim:
        fp = np.sum(p0 > tau) / len(p0)
        tp = np.sum(p1 > tau) / len(p1)
        s.loc[fp] = tp
    s = s.sort_index()

    # check for doubles
    assert len(s.index) == len(s.index.unique())

    # print auc
    print('Discriminator AUC: {}'.format(get_auc(s)))

    return s


def get_baserate(y, num_out, censored=False):
    if not censored:
        p = np.array([(y == i).mean() for i in range(num_out)])
        p = p[p > 0]
        return np.sum(p * np.log(p))
    else:
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


def count_loss(theta, y):
    # transformations
    pi = expit(theta[0])
    params = np.exp(theta[1:])
    a, b = params
    # zeros
    num_zeros = (y == 0).sum()
    lnl = num_zeros * np.log(pi + (1-pi) * a / (a + b))
    # non-zeros
    y1 = y[y > 0]
    lnl += len(y1) * (np.log(1-pi) + np.log(a) + loggamma(a + b) - loggamma(b))
    lnl += np.sum(loggamma(b + y1) - np.log(a + b + y1) - loggamma(a + b + y1))
    return -lnl


def main():
    d = dict()

    # loop over models, save training curves to dictionary
    for m in MODELS:
        print(m)

        # initialize dictionary
        key = 'bar_training_{}'.format(m)
        d[key] = pd.Series()

        # baserate
        y = load_inputs(TEST, m)['y']
        if m == BYR_HIST_MODEL:
            res = minimize(lambda theta: count_loss(theta, y),
                           x0=np.array([0., 0., 0.]),
                           method='Nelder-Mead')
            d[key]['Baserate'] = np.mean(-res.fun / len(y))
        else:
            num_out = NUM_OUT[m] if NUM_OUT[m] > 1 else 2
            d[key]['Baserate'] = get_baserate(y, num_out,
                                              censored=(m in CENSORED_MODELS))

        # test and training values
        _, lnl_test, lnl_train = extract_best_run(m)
        d[key]['Train'] = lnl_train[-1]
        d[key]['Test'] = lnl_test[-1]

        # likelihood
        d[key] = np.exp(d[key])

    # roc curve
    d['simple_roc'] = get_roc()

    # save output
    topickle(d, PLOT_DIR + 'training.pkl')


if __name__ == '__main__':
    main()
