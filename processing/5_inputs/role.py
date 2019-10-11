import sys, pickle, os, h5py
from compress_pickle import load, dump
import numpy as np, pandas as pd

sys.path.append('repo/')
from constants import *
from utils import *


def add_turn_indicators(df):
    indices = np.unique(df.index.get_level_values('index'))
    for i in range(len(indices)-1):
        ind = indices[i]
        featname = 't' + str((ind+1) // 2)
        df[featname] = df.index.isin([ind], level='index')
    return df


def parse_time_feats_role(model, outcome, x_offer):
    # initialize output dataframe
    idx = x_offer.index[x_offer.index.isin(IDX[model], level='index')]
    x_time = pd.DataFrame(index=idx)
    # current offer
    curr = x_offer.loc[idx]
    # last offer from other role
    offer1 = x_offer.groupby(['lstg', 'thread']).shift(
        periods=1).reindex(index=idx)
    # last offer from same role
    offer2 = x_offer.groupby(['lstg', 'thread']).shift(
        periods=2).reindex(index=idx)
    if model == 'byr':
        start = x_offer.xs(0, level='index')
        start = start.assign(index=1).set_index('index', append=True)
        offer2 = offer2.dropna().append(start).sort_index()
        # remove features that are constant for buyer
        offer2 = offer2.drop(['auto', 'exp', 'reject'], axis=1)
    else:
        offer1 = offer1.drop(['auto', 'exp', 'reject'], axis=1)
    # current offer
    excluded = ['round', 'nines', 'auto', 'exp', 'reject']
    if outcome in ['msg', 'con', 'accept', 'reject']:
        excluded += ['msg']
        if outcome in ['con', 'accept', 'reject']:
            excluded += ['con', 'norm', 'split']
    last_vars = [c for c in offer2.columns if c in excluded]
    # join dataframes
    x_time = x_time.join(curr.drop(excluded, axis=1))
    x_time = x_time.join(offer1.rename(
        lambda x: x + '_other', axis=1))
    x_time = x_time.join(offer2[last_vars].rename(
        lambda x: x + '_last', axis=1))
    # add turn indicators and return
    return add_turn_indicators(x_time)


# loads data and calls helper functions to construct training inputs
def process_inputs(part, model, outcome):
	# path name function
	getPath = lambda names: \
		'data/partitions/%s/%s.gz' % (part, '_'.join(names))

	# outcome
	y = load(getPath(['y', model, outcome])).astype('float32').unstack()
	y[y.isna()] = -1

	# fixed features
	x_fixed = load(getPath(['x', 'thread'])).join(cat_x_lstg(part))
	x_fixed = x_fixed.reindex(index=y.index)
	cols = list(x_fixed.columns)
	cols = cols[2:] + cols[:2]
	x_fixed = x_fixed[cols]

	# time features
	x_offer = load(getPath(['x', 'offer']))
	x_time = parse_time_feats_role(model, outcome, x_offer)

	return y, x_fixed, x_time


def get_sizes(outcome, y, x_fixed, x_time):
    sizes = {}
    # number of observations
    sizes['N'] = len(x_fixed.index)
    # fixed inputs
    sizes['fixed'] = len(x_fixed.columns)
    # output parameters
    if outcome == 'con':
        sizes['out'] = 3
    else:
        sizes['out'] = 1
   # RNN parameters
    sizes['steps'] = len(y.columns)
    sizes['time'] = len(x_time.columns)
    return sizes


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	parser.add_argument('--output', type=str, default='gz')
	num = parser.parse_args().num-1
	output = parser.parse_args().output

	# partition and outcome
	k = len(OUTCOMES_ROLE)
	part = PARTITIONS[num // (2 * k)]
	remainder = num % (2 * k)
	model = 'slr' if remainder >= k else 'byr'
	outcome = OUTCOMES_ROLE[remainder % k]
	outfile = lambda x: 'data/inputs/%s/%s_%s.pkl' % (x, model, outcome)
	print('Model: %s' % model)
	print('Outcome: %s' % outcome)
	print('Partition: %s' % part)

	# input dataframes, output processed dataframes
	y, x_fixed, x_time = process_inputs(part, model, outcome)

	# save featnames and sizes once
	if part == 'train_models':
		# save featnames
		featnames = {'x_fixed': x_fixed.columns, 'x_time': x_time.columns}
		pickle.dump(featnames, open(outfile('featnames'), 'wb'))

		# get data size parameters and save
		sizes = get_sizes(outcome, y, x_fixed, x_time)
		pickle.dump(sizes, open(outfile('sizes'), 'wb'))

	# convert to numpy
	data = {'y': y, 'x_fixed': x_fixed}
	data = {k: v.to_numpy().astype('float32') for k, v in data.items()}

	arrays = []
	for c in x_time.columns:
		array = x_time[c].astype('float32').unstack().reindex(
			index=y.index).to_numpy()
		arrays.append(np.expand_dims(array, axis=2))
	data['x_time'] = np.concatenate(arrays, axis=2)

	# save as either hdf5 or compressed numpy array
	path = 'data/inputs/%s/%s_%s.%s' % (part, model, outcome, output)
	if output == 'hdf5':
		f = h5py.File(path, 'w')
		for k, v in data.items():
			f.create_dataset(k, data=v, dtype='float32')
		f.close()
	else:
		dump(data, path)