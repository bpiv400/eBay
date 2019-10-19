import pandas as pd, numpy as np
from compress_pickle import dump


def add_turn_indicators(df):
    indices = np.unique(df.index.get_level_values('index'))
    for i in range(len(indices)-1):
        ind = indices[i]
        featname = 't' + str((ind+1) // 2)
        df[featname] = df.index.isin([ind], level='index')
    return df


def get_featnames(d):
	featnames = {'x_fixed': list(d['x_fixed'].columns)}
	if 'x_hour' in d:
		featnames['x_fixed'] += list(d['x_hour'].rename(
			lambda x: x + '_focal', axis=1).columns)
	if 'x_time' in d:
		featnames['x_time'] = list(d['x_time'].columns)
	return featnames


def get_sizes(d):
	sizes = {'N': len(d['y'].index), 
			 'fixed': len(d['x_fixed'].columns)}
	if 'x_hour' in d:
		sizes['fixed'] += len(d['x_hour'].columns)
	if 'x_time' in d:
		sizes['steps'] = len(d['y'].columns)
		sizes['time'] = len(d['x_time'].columns)
	return sizes


def convert_to_numpy(d):
	# y and x_fixed to numpy
	for k in ['y', 'x_fixed']:
		d[k] = d[k].to_numpy()

	# x_time to numpy
	if 'x_time' in d:
		arrays = []
		for c in x_time.columns:
			array = d['x_time'][c].astype('float32').unstack().reindex(
				index=y.index).to_numpy()
			arrays.append(np.expand_dims(array, axis=2))
		d['x_time'] = np.concatenate(arrays, axis=2)

	return d