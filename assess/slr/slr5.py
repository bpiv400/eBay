import numpy as np
from assess.util import ll_wrapper, add_byr_reject_on_lstg_expiration
from utils import topickle, load_data
from assess.const import NORM1_DIM
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, TEST, EXP, AUTO, NORM, REJECT


def main():
    d = dict()

    data = load_data(part=TEST)
    con = add_byr_reject_on_lstg_expiration(con=data[X_OFFER][CON])
    norm = data[X_OFFER][NORM]

    rej = data[X_OFFER][REJECT] & ~data[X_OFFER][AUTO] & ~data[X_OFFER][EXP]
    rej2 = rej.xs(2, level=INDEX)
    idx = rej2[rej2].index

    con5 = con.xs(5, level=INDEX)
    idx = idx.intersection(con5.index)
    con5 = con5.loc[idx]

    norm3 = norm.xs(3, level=INDEX).loc[con5.index]

    offers4 = data[X_OFFER][[CON, AUTO, EXP]].xs(4, level=INDEX).loc[con5.index]
    con4 = offers4[CON] - .05 * offers4[AUTO] - .1 * offers4[EXP]
    assert con4.min() == -.1

    x1, x2, y = norm3.values, con4.values, (con5 == 1).values

    key = 'response_rejrejacc'
    mask = np.isclose(x2, 0) & (x1 > .33)
    line, bw = ll_wrapper(y[mask], x1[mask], dim=NORM1_DIM)
    print(bw[0])
    d[key] = line

    topickle(d, PLOT_DIR + 'slr5.pkl')


if __name__ == '__main__':
    main()
