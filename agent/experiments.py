from itertools import product
import pandas as pd
from constants import RL_LOG_DIR


ranges = dict(entropy_coeff=[.005, .01, .05, .1])


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
