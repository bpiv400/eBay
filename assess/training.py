import numpy as np
import pandas as pd
from sim.best_models import extract_best_run
from sim.EBayDataset import EBayDataset
from utils import load_inputs, topickle, get_model_predictions
from inputs.const import NUM_OUT
from constants import PLOT_DIR
from featnames import TEST, MODELS, CENSORED_MODELS, DISCRIM_MODEL


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


def main():
    d = dict()

    # loop over models, save training curves to dictionary
    for m in MODELS:
        print(m)

        # number of periods
        num_out = NUM_OUT[m] if NUM_OUT[m] > 1 else 2

        # find best performing experiment
        run, lnl_test, lnl_train = extract_best_run(m)

        # initialize dictionary
        key = 'training_{}'.format(m)
        d[key] = pd.DataFrame(index=range(len(lnl_test)+1))

        # load data
        y = load_inputs(TEST, m)['y']

        # initialization value
        if m in CENSORED_MODELS:
            p_arrival = np.ones((y >= 0).sum()) / num_out
            p_cens = -y[y < 0] / num_out
            lnl0 = np.log(np.concatenate([p_arrival, p_cens], axis=0)).mean()
        else:
            lnl0 = np.log(1 / num_out)

        # training curves, with initialization value
        d[key]['test'] = [lnl0] + lnl_test
        d[key]['train'] = [lnl0] + lnl_train

        # baserate
        d[key]['baserate'] = get_baserate(y, num_out,
                                          censored=(m in CENSORED_MODELS))

        # exponentiate
        d[key] = np.exp(d[key])

    # roc curve
    d['simple_roc'] = get_roc()

    # save output
    topickle(d, PLOT_DIR + 'training.pkl')


if __name__ == '__main__':
    main()
