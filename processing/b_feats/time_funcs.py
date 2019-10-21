import numpy as np
import pandas as pd


# cumulative number of offers made by role
def num_offers(df, role):
    if role == 'slr':
        s = ~df.byr & ~df.reject
    elif role == 'byr':
        s = df.byr & ~df.reject
    total = s.groupby('lstg').cumsum()
    focal = s.groupby(['lstg', 'thread']).cumsum()
    return (total - focal).astype(np.int64)


# number of open listings
def open_lstgs(df, levels):
    # variables from index levels
    thread = df.index.get_level_values('thread')
    index = df.index.get_level_values('index')
    # listing opens at thread 0 and index 0
    start = pd.Series(index == 0, index=df.index)
    # listing ends with accept or expiration
    end = df.accept | ((thread == 0) & (index == 1))
    # open - closed
    s = start.astype(np.int64) - end.astype(np.int64)
    # cumulative total by levels grouping
    return s.groupby(by=levels).cumsum()


# number of threads for which an offer from role is outstanding
def open_offers(df, levels, role):
    # index number
    if 'index' in df.columns:
        index = df['index']
    else:
        index = df.index.get_level_values('index')
    # open and closed markers
    if role == 'slr':
        start = ~df.byr & ~df.accept & (index > 0)
        end = df.byr & (index > 1)
    elif role == 'byr':
        start = df.byr & ~df.reject & ~df.accept
        end = ~df.byr & (index > 1)
        # print(start)
        # print(end)
    # open - closed
    s = start.astype(np.int64) - end.astype(np.int64)
    # cumulative sum by levels grouping
    return s.groupby(by=levels).cumsum()


# best offer from role
def past_best(df, role, levels='lstg', index=None):
    s = df.norm.copy()
    if role == 'slr':
        s.loc[df.byr] = 0.0
    elif role == 'byr':
        s.loc[~df.byr] = 0.0
        s.loc[s.isna()] = 0.0
    if index is not None:
        s.loc[index] = 0.0
    return s.groupby(by=levels).cummax()