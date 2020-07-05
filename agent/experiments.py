from itertools import product
import pandas as pd
from constants import RL_LOG_DIR


ranges = dict(batch_size=[4096],
              lr_step_batches=[16, 32],
              ratio_clip=[.1, .2],
              clip_grad_norm=[-1, 0.5, 1],
              entropy_coeff=[.01, .05],
              lr=[1e-3, 1e-4],
              same_lr=[True])


def main():
    # create dataframe from combinations
    combs = list(product(*list(ranges.values())))
    exps = pd.DataFrame.from_records(combs,
                                     columns=ranges.keys(),
                                     index=pd.Index(range(len(combs)),
                                                    name='exp_id'))

    # save experiments file
    exps.to_csv(RL_LOG_DIR + 'exps.csv')


if __name__ == '__main__':
    main()
