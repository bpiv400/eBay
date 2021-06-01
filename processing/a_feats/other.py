import numpy as np
import pandas as pd
from utils import topickle, load_feats
from constants import FEATS_DIR, START, END, MAX_DAYS, HOLIDAYS, NUM_COMMON_CONS
from featnames import HOLIDAY, DOW_PREFIX, CON, INDEX


def save_date_feats():
    N = pd.to_datetime(END) - pd.to_datetime(START) \
        + pd.to_timedelta(MAX_DAYS, unit='d') \
        + pd.to_timedelta(1, unit='s')
    days = pd.to_datetime(list(range(N.days)), unit='D', origin=START)
    df = pd.DataFrame(index=days)
    df[HOLIDAY] = days.isin(HOLIDAYS)
    for i in range(6):
        df[DOW_PREFIX + str(i)] = days.dayofweek == i
    topickle(df.values, FEATS_DIR + 'date_feats.pkl')


def restrict_cons(con, turns=None):
    s = con[con.index.isin(turns, level=INDEX)]
    pdf = s.groupby(s).count().sort_values() / len(s)
    cons = pdf.index.values[-NUM_COMMON_CONS:]
    return np.sort(cons)


def save_common_cons():
    con = load_feats('offers')[CON]
    con = con[(con > 0) & (con < 1)]
    d = dict()
    d[1] = restrict_cons(con, turns=[1])
    d[3] = d[5] = restrict_cons(con, turns=[3, 5])
    d[2] = d[4] = d[6] = restrict_cons(con, turns=[2, 4, 6])
    topickle(d, FEATS_DIR + 'common_cons.pkl')


def main():
    save_date_feats()
    save_common_cons()


if __name__ == "__main__":
    main()
