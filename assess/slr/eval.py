from assess.util import get_eval_df
from utils import topickle
from agent.const import DELTA_SLR
from constants import PLOT_DIR
from featnames import SLR


def main():
    d = dict()

    # seller
    for delta in DELTA_SLR:
        df = get_eval_df(byr=True, delta=delta)

        for c in df.columns:
            d['bar_{}_{}'.format(c, delta)] = df[c]

    topickle(d, PLOT_DIR + '{}_eval.pkl'.format(SLR))


if __name__ == '__main__':
    main()
