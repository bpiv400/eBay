from assess.util import get_eval_df
from utils import topickle
from agent.const import DELTA_BYR
from constants import PLOT_DIR
from featnames import BYR


def main():
    d = dict()

    for delta in DELTA_BYR:
        df = get_eval_df(byr=True, delta=delta)

        for c in df.columns:
            d['bar_{}_{}'.format(c, delta)] = df[c]

    topickle(d, PLOT_DIR + '{}eval.pkl'.format(BYR))


if __name__ == '__main__':
    main()
