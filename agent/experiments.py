from itertools import product
from compress_pickle import dump
import pandas as pd
from constants import RL_LOG_DIR


ranges = dict(batch_size=[2 ** 12, 2 ** 14, 2 ** 16],
              patience=[0, 1, 2],
              ratio_clip=[0., .1, .2],
              clip_grad_norm=[-1, 0.5, 1],
              entropy_coeff=[1, .1],
              lr=[1e-3, 1e-4],
              same_lr=[True, False])


def main():
    # create dataframe from combinations
    combs = list(product(*list(ranges.values())))
    exps = pd.DataFrame.from_records(combs, columns=ranges.keys())

    # save experiments file
    exp_path = RL_LOG_DIR + 'exps.pkl'
    dump(exps, exp_path)


if __name__ == '__main__':
    main()
