import numpy as np, pandas as pd

LEVELS = ['slr', 'meta', 'leaf', 'cndtn', 'title', 'lstg']
FUNCTIONS = {'slr': [open_lstgs, open_threads],
             'meta': [open_lstgs, open_threads],
             'leaf': [open_lstgs, open_threads],
             'cndtn': [slr_min, byr_max,
                       slr_offers, byr_offers,
                       open_lstgs, open_threads],
             'title': [slr_min, byr_max,
                       slr_offers, byr_offers,
                       open_lstgs, open_threads],
             'lstg': [slr_min, byr_max,
                      slr_offers, byr_offers,
                      open_threads]}

# minimum offer made by a slr
def slr_min(df, levels):
    keep = df.index.get_level_values('index') % 2 == 0
    df.loc[~keep, 'price'] = np.inf
    return df['price'].groupby(by=levels).cummin()


# maximum offer made by a byr
def byr_max(df, levels):
    keep = df.index.get_level_values('index') % 2 == 1
    df.loc[~keep, 'price'] = 0
    return df['price'].groupby(by=levels).cummax()


# cumulative number of offers made by byrs
def byr_offers(df, levels):
    idx = df.index.get_level_values('index')
    s = (df['price'] >= 0) & (idx % 2 == 1) & ~df['accept'] & ~df['reject']
    return s.astype(np.int64).groupby(by=levels).cumsum()


# cumulative number of offers made by the slr
def slr_offers(df, levels):
    thread = df.index.get_level_values('thread')
    idx = df.index.get_level_values('index')
    s = (thread > 0) & (idx % 2 == 0) & ~df['accept'] & ~df['reject']
    return s.astype(np.int64).groupby(by=levels).cumsum()


# number of other threads for which an offer is outstanding
def open_threads(df, levels):
    idx = df.index.get_level_values('index')
    s = (idx % 2 == 1).astype(np.int64) - df['accept'] - df['reject']
    total = s.groupby(by=levels).cumsum()
    focal = s.groupby(by=['lstg', 'thread']).cumsum()
    diff = total - focal
    return diff.groupby(by=levels + ['clock']).transform('max')


# number of open listings
def open_lstgs(df, levels):
    thread = df.index.get_level_values('thread')
    idx = df.index.get_level_values('index')
    opened = (df['price'] > 0) & (thread == 0) & (idx == 0)
    closed = ((df['price'] > 0) & (thread == 0) & (idx == 1)) | (df['accept'] == 1)
    s = opened.astype(np.int64) - closed.astype(np.int64)
    runmax = s.groupby(by=levels).cumsum().rename('runmax')
    return runmax.groupby(by=levels + ['clock']).transform('max')


def add_time_feats(events):
    time_feats = pd.DataFrame(index=events.index)
    for i in range(len(LEVELS)):
        levels = LEVELS[: i+1]
        ordered = events.copy().sort_values(levels + ['clock'])
        name = levels[-1]
        f = FUNCTIONS[name]
        for j in range(len(f)):
            feat = f[j](ordered.copy(), levels)
            if f[j].__name__ in ['slr_min', 'byr_max']:
                feat = feat.div(L['start_price'])
                feat = feat.reorder_levels(
                    LEVELS + ['thread', 'index', 'clock'])
            newname = '_'.join([name, f[j].__name__])
            print('\t%s' % newname)
            time_feats[newname] = feat
    time_feats.reset_index('clock', drop=True, inplace=True)
    return events.reset_index('clock').join(time_feats)
