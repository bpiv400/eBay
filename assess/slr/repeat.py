import pandas as pd
from agent.util import load_valid_data
from assess.util import ll_wrapper
from utils import topickle, load_feats
from assess.const import NORM1_DIM
from constants import PLOT_DIR, DAY
from featnames import X_OFFER, CON, INDEX, AUTO, CLOCK, BYR, SLR, LOOKUP, \
    LSTG, THREAD


def main():
    # load data
    data = load_valid_data(byr=False, clock=True)

    # running variable
    con2 = data[X_OFFER].loc[~data[X_OFFER][AUTO], CON].xs(2, level=INDEX)
    x = data[X_OFFER][CON].xs(1, level=INDEX).loc[con2.index].values

    # roles
    byr = load_feats('threads', lstgs=data[LOOKUP].index)[BYR]
    slr = load_feats('listings', lstgs=data[LOOKUP].index)[SLR]
    arrival = data[CLOCK].xs(1, level=INDEX)
    roles = pd.concat([byr, arrival], axis=1).join(slr).reset_index().sort_values(
        [SLR, BYR, CLOCK]).set_index([SLR, BYR, LSTG, THREAD]).squeeze()

    # time to next arrival
    count = roles.groupby([SLR, BYR]).count()
    diff0 = pd.Series(index=roles[count == 1].index, name=CLOCK)
    diff1 = roles[count > 1].groupby([SLR, BYR]).diff()
    diff = pd.concat([diff0, diff1]).reset_index(
        [SLR, BYR], drop=True).sort_index().loc[con2.index]
    y = (diff < 7 * DAY).values

    # estimate three lines
    mask = {'Pr(accept)': con2 == 1,
            'Pr(reject)': con2 == 0,
            'Pr(counter)': (con2 > 0) & (con2 < 1)}

    df = pd.DataFrame()
    for k, v in mask.items():
        line, bw = ll_wrapper(y=y[v], x=x[v], dim=NORM1_DIM)
        line.columns = pd.MultiIndex.from_product([[k], line.columns])
        df = pd.concat([df, line], axis=1)
        print('{}: {}'.format(k, bw[0]))

    # save
    d = {'simple_conrepeat': df}
    topickle(d, PLOT_DIR + 'slrrepeat.pkl')


if __name__ == '__main__':
    main()
