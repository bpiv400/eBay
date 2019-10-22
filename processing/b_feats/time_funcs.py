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
    converter = df['clock'].xs(0, level='thread').xs(0, level='index').reset_index()

    # listing opens at thread 0 and index 0
    start_end_df = pd.DataFrame(data={'clock': df['clock'],
                               'start': index == 0,
                               'end': df.accept | ((thread == 0) & (index == 1))},
                         index=df.index)

    # experimental
    start_end_df = start_end_df.groupby(by=levels + ['clock']).sum()
    start = start_end_df['start']
    end = start_end_df['end']
    s = start.astype(np.int64) - end.astype(np.int64)
    s = s.groupby(by=levels).cumsum().to_frame(name='lstg_open').reset_index()
    s = s.merge(converter, on=levels + ['clock'], how='inner')
    s = s.set_index(levels + ['lstg'], append=False, drop=True)
    s = s['lstg_open']
    # cumulative total by levels grouping
    return s


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