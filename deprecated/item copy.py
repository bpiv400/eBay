
LEVELS = ['slr', 'meta', 'leaf', 'title', 'cndtn', 'lstg']


df = L[LEVELS[:-1] + ['start_date', 'end_time']].reset_index().set_index(
	LEVELS[:-1]).reorder_levels(LEVELS[:-1]).sort_index()
df['start_date'] *= 24 * 3600
df = df.rename(lambda x: x.split('_')[0], axis=1)

# find multi-listings
df = df.sort_values(df.index.names + ['start'])
maxend = df.end.groupby(df.index.names).cummax()
maxend = maxend.groupby(df.index.names).shift(1)
overlap = df.start <= maxend
ismulti = overlap.groupby(df.index.names).max()

# drop multi-listings
df = df[~multi]


df['connected'] = df.start == df.end.groupby('item').shift(1)+1

df['count'] = df.lstg.groupby(df.index.names).transform('count')

multi = df[df['count'] > 1]