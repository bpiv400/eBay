import sys, pickle, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def parse_time_feats_role(role, x_offer):
	# initialize output dataframe
	idx = x_offer.index[x_offer.index.isin(IDX[role], level='index')]
	x_time = pd.DataFrame(index=idx)
	# current offer
	curr = x_offer.loc[idx]
	# last offer from other role
	offer1 = x_offer.groupby(['lstg', 'thread']).shift(
		periods=1).reindex(index=idx)
	# last offer from same role
	offer2 = x_offer.groupby(['lstg', 'thread']).shift(
		periods=2).reindex(index=idx)
	if role == 'byr':
		start = x_offer.xs(0, level='index')
		start = start.assign(index=1).set_index('index', append=True)
		offer2 = offer2.dropna().append(start).sort_index()
		# remove features that are constant for buyer
		offer2 = offer2.drop(['auto', 'exp', 'reject'], axis=1)
	else:
		offer1 = offer1.drop(['auto', 'exp', 'reject'], axis=1)
	# current offer
	excluded = ['auto', 'exp', 'reject', 'con', 'norm', 'split']
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
def process_inputs(part, role):
	# path name function
	getPath = lambda names: '%s/partitions/%s/%s.gz' % \
		(PREFIX, part, '_'.join(names))

	# outcome
	y = load(getPath(['y', 'con', role])).astype(
		'float32').unstack(fill_value=-1)

	# x_fixed: x_lstg and x_thread
	x_fixed = load(getPath(['x', 'lstg'])).reindex(
		index=y.index, level='lstg').join(load(getPath(['x', 'thread'])))

	# time features
	x_offer = load(getPath(['x', 'offer']))
	raw = [c for c in x_offer.columns if c.endswith('_raw')]
	x_offer = x_offer.drop(raw, axis=1)
	x_time = parse_time_feats_role(role, x_offer)

	return {'y': y.astype('float32', copy=False), 
			'x_fixed': x_fixed.astype('float32', copy=False), 
			'x_time': x_time.astype('float32', copy=False)}


if __name__ == '__main__':
	# extract model from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1

	# partition and role
	part = PARTITIONS[num // 2]
	role = 'slr' if num % 2 else 'byr'
	model = 'con_%s' % role
	print('%s/%s' % (part, model))

	# input dataframes, output processed dataframes
	d = process_inputs(part, role)

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/con_%s.pkl' % (PREFIX, role), 'wb'))

		sizes = get_sizes(d)
		sizes['dim'] = np.arange(0, 1.01, 0.01)
		pickle.dump(sizes, 
			open('%s/inputs/sizes/con_%s.pkl' % (PREFIX, role), 'wb'))

	# save dictionary of numpy arrays
	dump(convert_to_numpy(d), 
		'%s/inputs/%s/con_%s.gz' % (PREFIX, part, role))