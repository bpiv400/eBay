from agent.util import get_run_dir, load_valid_data, get_slr_valid
from assess.util import kreg2
from utils import topickle, load_data
from agent.const import DELTA_SLR
from assess.const import NORM1_DAYS_MESH
from constants import PLOT_DIR
from featnames import X_OFFER, REJECT, INDEX, TEST, NORM, AUTO, X_THREAD, \
    THREAD, DAYS_SINCE_LSTG


def get_feats(data=None):
    df = data[X_OFFER].loc[~data[X_OFFER][AUTO], [REJECT, NORM]]
    rej2 = df[REJECT].xs(2, level=INDEX).xs(1, level=THREAD)
    norm1 = df[NORM].xs(1, level=INDEX).xs(1, level=THREAD).loc[rej2.index]
    days = data[X_THREAD][DAYS_SINCE_LSTG].xs(1, level=THREAD).loc[rej2.index]
    # throw out extreme values (helps with bandwidth estimation)
    mask = (norm1 > .33) & (norm1 < .9) & (days < 5)
    rej2, norm1, days = rej2[mask], norm1[mask], days[mask]
    return rej2.values, norm1.values, days.values


def days_plot(y=None, x1=None, x2=None, bw=None):
    mask = (x1 < .9) & (x2 < 3)
    s, bw = kreg2(y=y[mask], x1=x1[mask], x2=x2[mask],
                  mesh=NORM1_DAYS_MESH, bw=bw)
    print('days: {}'.format(bw))
    return s, bw


def main():
    d, bw = dict(), None

    # observed sellers
    data = get_slr_valid(load_data(part=TEST))
    y, x1, x2 = get_feats(data=data)
    d['contour_rejdays_data'], bw = days_plot(y=y, x1=x1, x2=x2)

    # seller runs
    for delta in DELTA_SLR[:-1]:
        run_dir = get_run_dir(byr=False, delta=delta)
        data = load_valid_data(part=TEST, run_dir=run_dir)
        y, x1, x2 = get_feats(data=data)
        d['contour_rejdays_{}'.format(delta)], _ = \
            days_plot(y=y, x1=x1, x2=x2, bw=bw)

    topickle(d, PLOT_DIR + 'slr2days.pkl')


if __name__ == '__main__':
    main()
