import pandas as pd
from compress_pickle import dump
from constants import INPUT_DIR, START, END, MAX_DAYS, HOLIDAYS
from featnames import HOLIDAY, DOW_PREFIX

# clock features by second
N = pd.to_datetime(END) - pd.to_datetime(START) + pd.to_timedelta(MAX_DAYS, unit='d')
days = pd.to_datetime(range(N.days), unit='D', origin=START)
df = pd.DataFrame(index=days)
df[HOLIDAY] = days.isin(HOLIDAYS)
for i in range(6):
    df[DOW_PREFIX + str(i)] = days.dayofweek == i
dump(df.values, INPUT_DIR + 'date_feats.pkl')
