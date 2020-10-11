import argparse
import os
from plots.util import count_plot, action_plot, cdf_plot, diag_plot, \
    response_plot, contour_plot, w2v_plot, pdf_plot
from utils import unpickle
from constants import PLOT_DIR, FIG_DIR

PLOT_TYPES = {'num_threads': count_plot,
              'num_offers': count_plot,
              'accept': contour_plot,
              'response': response_plot,
              'norm-norm': diag_plot,
              'action': action_plot,
              'normval': contour_plot,
              'w2v': w2v_plot,
              'cdf': cdf_plot,
              'pdf': pdf_plot}


def main():
    # subset
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--subset', type=str)
    args = parser.parse_args()
    prefix, subset = args.prefix, args.subset

    path = PLOT_DIR + prefix
    folder = '{}/'.format(prefix)
    if subset is not None:
        path += '_{}'.format(subset)
        folder += '{}/'.format(subset)

    if not os.path.isdir(FIG_DIR + folder):
        os.makedirs(FIG_DIR + folder)

    d = unpickle('{}.pkl'.format(path))
    for k, v in d.items():
        prefix = k.split('_')[0]
        f = PLOT_TYPES[prefix]
        if type(v) is not dict:
            path = folder + k
            print(path)
            f(path, v)
        elif k == 'accept':
            for t, d_k in v.items():
                for key, s in d_k.items():
                    path = folder + '{}_{}_{}'.format(k, key, t)
                    print(path)
                    f(path, s)
        else:
            for t, df in v.items():
                path = folder + '{}_{}'.format(k, t)
                print(path)
                f(path, df)


if __name__ == '__main__':
    main()
