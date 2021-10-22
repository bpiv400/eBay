import numpy as np
from agent.util import load_valid_data
from analyze.util import ll_wrapper, save_dict
from analyze.const import POINTS
from featnames import X_OFFER, X_THREAD, BYR_HIST, CON

DIM = np.linspace(0, 3, POINTS)
LOG10ZERO = -np.log10(2)


def main():
    d = dict()

    # average buyer experience by first offer
    data = load_valid_data(byr=True, minimal=True)

    # log experience
    loghist = np.log10(data[X_THREAD][BYR_HIST])
    loghist[loghist.isna()] = LOG10ZERO

    # outcomes
    con = data[X_OFFER][CON].unstack()
    acc1 = con[1] == 1
    con1 = con.loc[~acc1, 1]
    offers = (con > 0) & (con < 1)
    offers = offers[[1, 3, 5]].sum(axis=1)
    offers = offers[offers > 0]
    assert np.all(con1.index == offers.index)

    # local linear regression
    for outcome in ['acc1', 'con1', 'offers']:
        x = loghist.loc[locals()[outcome].index].values
        y = locals()[outcome].values
        line, dots, bw = ll_wrapper(y=y, x=x, discrete=[LOG10ZERO], dim=DIM)
        print('{}: {}'.format(outcome, bw[0]))
        d['response_hist{}'.format(outcome)] = line, dots

    # save
    save_dict(d, 'byrhist')


if __name__ == '__main__':
    main()
