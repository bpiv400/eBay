import pandas as pd, numpy as np
from compress_pickle import dump
import h5py


def add_turn_indicators(df):
    indices = np.unique(df.index.get_level_values('index'))
    for i in range(len(indices)-1):
        ind = indices[i]
        featname = 't' + str((ind+1) // 2)
        df[featname] = df.index.isin([ind], level='index')
    return df


def save_params_data(path, part, d):
	if part == 'train_models':
		# construct featnames and sizes
		featnames = {'x_fixed': list(d['x_fixed'].columns)}
		sizes = {'N': len(d['y'].index), 
				 'fixed': len(d['x_fixed'].columns)}
		if 'arrival' in path:
			featnames['x_fixed'] += list(d['x_period'].rename(
				lambda x: x + '_focal', axis=1).columns)
			sizes['fixed'] += len(d['x_days'].columns)
		if 'x_time' in d:
			featnames['x_time'] = list(d['x_time'].columns)
			sizes['steps'] = len(d['y'].columns)
    		sizes['time'] = len(d['x_time'].columns)

		# save featnames and zies
		pickle.dump(featnames, open(path('featnames') + '.pkl', 'wb'))
		pickle.dump(sizes, open(path('sizes') + '.pkl', 'wb'))

	# y and x_fixed to numpy
	data = {}
	for k in ['y', 'x_fixed']:
		data[k] = d[k].to_numpy()

	# x_time to numpy
	if 'x_time' in d:
		arrays = []
		for c in x_time.columns:
			array = x_time[c].astype('float32').unstack().reindex(
				index=y.index).to_numpy()
			arrays.append(np.expand_dims(array, axis=2))
		data['x_time'] = np.concatenate(arrays, axis=2)

	# save as hdf5
	f = h5py.File(path('inputs') + '.hdf5', 'w')
	for k, v in data.items():
		f.create_dataset(k, data=v, dtype=v.dtype)
	f.close()