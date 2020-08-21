import argparse
import os
from plots.util import count_plot, action_plot, dist_plot, diag_plot, contour_plot
from utils import unpickle
from constants import PLOT_DIR, FIG_DIR


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
        if type(v) is not dict:
            f = count_plot if k.startswith('num') else dist_plot
            path = folder + k
            print(path)
            f(path, v)
        elif k == 'accept':
            for t, d_k in v.items():
                for key, s in d_k.items():
                    path = folder + '{}_{}_{}'.format(k, key, t)
                    print(path)
                    contour_plot(path, s)
        else:
            if k == 'action':
                f = action_plot
            elif k == 'norm-norm':
                f = diag_plot
            else:
                f = dist_plot
            for t, df in v.items():
                path = folder + '{}_{}'.format(k, t)
                print(path)
                f(path, df)


if __name__ == '__main__':
    main()
