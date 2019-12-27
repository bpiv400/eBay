import numpy as np, pandas as pd
from compress_pickle import dump
from constants import INPUT_DIR, START, END, MAX_DAYS
from processing.processing_utils import extract_day_feats

# clock features by second
N = pd.to_datetime(END) - pd.to_datetime(START) + pd.to_timedelta(MAX_DAYS, unit='d')
days = pd.to_datetime(range(N.days), unit='D', origin=START)
df = extract_day_feats(pd.Series(days))
dump(df.values, INPUT_DIR + 'date_feats.pkl')