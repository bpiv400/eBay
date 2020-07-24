from compress_pickle import load
from plots.util import roc_plot
from constants import PLOT_DIR, DISCRIM_MODEL


def get_auc(s):
    fp = s.index.values
    fp_delta = fp[1:] - fp[:-1]
    tp = s.values
    tp_bar = (tp[1:] + tp[:-1]) / 2
    auc = (fp_delta * tp_bar).sum()
    return auc


def main():
    # load data
    s = load(PLOT_DIR + 'roc.pkl')

    # auc
    print('AUC: {}'.format(get_auc(s)))

    # roc plot
    roc_plot(DISCRIM_MODEL, s)


if __name__ == '__main__':
    main()
