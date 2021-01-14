import argparse
import os
import numpy as np
import pandas as pd
import plots.util as util
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
    if type(d) is pd.DataFrame:
        print(np.abs(d).mean())
    else:
        for k, v in d.items():
            prefix = k.split('_')[0]
            f_name = '{}_plot'.format(prefix)
            f = getattr(util, f_name)
            if type(v) is not dict or prefix == 'slr':
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
