import pandas as pd
from assess.byr.util import get_hist_bins
from assess.util import ll_wrapper, add_byr_reject_on_lstg_expiration
from processing.util import feat_to_pctile
from utils import load_data, topickle
from assess.const import NORM1_DIM
from constants import PLOT_DIR
from featnames import X_OFFER, X_THREAD, CON, INDEX, TEST, EXP, AUTO, NORM, REJECT, \
    BYR_HIST


def main():
    data = load_data(part=TEST)
    con = add_byr_reject_on_lstg_expiration(con=data[X_OFFER][CON])

    # outcome: buyer accepts in turn 3
    acc3 = con.xs(3, level=INDEX) == 1

    # running variable: buyer's first offer
    norm1 = data[X_OFFER][NORM].xs(1, level=INDEX)

    # master index: seller actively rejects in turn 2
    df2 = data[X_OFFER][[REJECT, AUTO, EXP]].xs(2, level=INDEX)
    idx = df2[df2[REJECT] & ~df2[AUTO] & ~df2[EXP]].index.intersection(acc3.index)
    y, x = acc3.loc[idx].values, norm1.loc[idx].values

    # buyer experience
    hist = feat_to_pctile(s=data[X_THREAD][BYR_HIST].loc[idx], reverse=True)
    start, labels = get_hist_bins()

    # local linear estimate, using bandwidth from hist = 0
    bw, df = None, pd.DataFrame()
    for i in range(len(start)):
        if i < len(start) - 1:
            mask = (hist >= start[i]) & (hist < start[i+1])
        else:
            mask = hist >= start[i]
        if bw is None:
            line, bw = ll_wrapper(y=y[mask], x=x[mask], dim=NORM1_DIM, ci=False)
            print('bw: {}'.format(bw[0]))
        else:
            line, _ = ll_wrapper(y=y[mask], x=x[mask], dim=NORM1_DIM, ci=False, bw=bw)
        df[labels[i]] = line

    # put in dictionary and save
    d = {'simple_rejacchist': df}
    topickle(d, PLOT_DIR + 'slr3.pkl')
