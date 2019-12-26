import numpy as np, pandas as pd
from compress_pickle import dump
from constants import PREFIX, START, END, MAX_DAYS
from utils import extract_clock_feats


# clock features by second
N = pd.to_datetime(END) - pd.to_datetime(START) + pd.to_timedelta(MAX_DAYS, unit='d')
sec = pd.to_datetime(range(int(N.total_seconds())), unit='s', origin=START)
df = extract_clock_feats(pd.Series(sec))
dump(df.values, '{}/inputs/universal/x_clock.gz'.format(PREFIX))