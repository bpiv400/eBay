from itertools import product
import pandas as pd
from constants import AGENT_DIR

ranges = dict(entropy_coeff=[.001, .005],
              lr=[.0005, .0001],
              ratio_clip=[.1, .2])


def main():
    # create dataframe from combinations
    combs = list(product(*list(ranges.values())))
    exps = pd.DataFrame.from_records(combs,
                                     columns=ranges.keys(),
                                     index=pd.Index(range(len(combs)),
                                                    name='exp_id'))

    # save experiments file
    exps.to_csv(AGENT_DIR + 'exps.csv')


if __name__ == '__main__':
    main()
