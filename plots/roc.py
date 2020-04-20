from compress_pickle import load
from plots.plots_utils import roc_plot
from constants import PLOT_DIR


def get_auc(s):
    fp = s.index.values
    fp_delta = fp[1:] - fp[:-1]
    tp = s.values
    tp_bar = (tp[1:] + tp[:-1]) / 2
    auc = (fp_delta * tp_bar).sum()
    return auc


def main():
    # load data
    d = load(PLOT_DIR + 'roc.pkl')

    for m, s in d.items():
        # auc
        print('{}: {}'.format(m, get_auc(s)))

        # roc plot
        roc_plot(m, d[m])


if __name__ == '__main__':
    main()
