import numpy as np
import pandas as pd
from agent.util import only_byr_agent, load_valid_data
from processing.util import feat_to_pctile
from utils import topickle, load_pctile
from constants import PLOT_DIR
from featnames import X_OFFER, X_THREAD, BYR_HIST, CON


def get_hist_bins(num=6):
    s = load_pctile(name=BYR_HIST)
    start = []
    for i in range(num):
        idx = np.searchsorted(s, i / num)
        start.append(s.index[idx])
    start = np.unique(start)
    labels = []
    for i in range(len(start) - 1):
        first, last = start[i], start[i+1]-1
        label = '{}'.format(first)
        if last > first:
            label += '-{}'.format(last)
        labels.append(label)
    labels.append('{}+'.format(start[-1]))
    return start, labels


def main():
    d = dict()

    # first thread in data
    data = only_byr_agent(load_valid_data(byr=True, minimal=True))

    # experience
    hist = feat_to_pctile(data[X_THREAD][BYR_HIST], reverse=True)
    start, labels = get_hist_bins()

    # outcomes
    con = data[X_OFFER][CON].unstack()
    acc1 = con[1] == 1
    con1 = con.loc[~acc1, 1]
    offers = (con > 0) & (con < 1)
    offers = offers[[1, 3, 5]].sum(axis=1)
    offers = offers[offers > 0]
    assert np.all(con1.index == offers.index)

    # averages by experience bin
    df = pd.DataFrame()
    for outcome in ['acc1', 'con1', 'offers']:
        s = locals()[outcome]
        x = hist.loc[s.index].values
        y = s.values
        for i in range(len(start)):
            first = start[i]
            if i < len(start) - 1:
                last = start[i+1] - 1
                idx = (x >= first) & (x <= last)
            else:
                idx = x >= first
            y_i = y[idx]
            beta, se = y_i.mean(), y_i.std() / np.sqrt(len(y_i))
            df.loc[labels[i], outcome] = y[idx].mean()

        d['coef_hist{}'.format(outcome)] = df

    # save
    topickle(d, PLOT_DIR + 'byrhist.pkl')


if __name__ == '__main__':
    main()
