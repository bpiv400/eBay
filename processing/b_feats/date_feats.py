import pandas as pd
from utils import topickle
from constants import DATE_FEATS_PATH, START, END, MAX_DAYS, HOLIDAYS
from featnames import HOLIDAY, DOW_PREFIX


def main():
    N = pd.to_datetime(END) - pd.to_datetime(START) \
        + pd.to_timedelta(MAX_DAYS, unit='d') \
        + pd.to_timedelta(1, unit='s')
    days = pd.to_datetime(list(range(N.days)), unit='D', origin=START)
    df = pd.DataFrame(index=days)
    df[HOLIDAY] = days.isin(HOLIDAYS)
    for i in range(6):
        df[DOW_PREFIX + str(i)] = days.dayofweek == i
    topickle(df.values, DATE_FEATS_PATH)


if __name__ == "__main__":
    main()
