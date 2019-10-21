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
def process_inputs(part, model):
	role = model.split('_')[1]

	# path name function
	getPath = lambda names: '%s/partitions/%s/%s.gz' % \
		(PREFIX, part, '_'.join(names))

	# outcome
	y = load(getPath(['y', model])).astype('float32').unstack()
	y[y.isna()] = -1

	# fixed features
	x_thread = load(getPath(['x', 'thread']))
	x_fixed = cat_x_lstg(part).reindex(index=y.index).join(x_thread)

	# time features
	x_offer = load(getPath(['x', 'offer']))
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

	# out path
	path = lambda x: '%s/%s/%s/con_%s.gz' % (PREFIX, x, part, role)

	# input dataframes, output processed dataframes
	d = process_inputs(part, model)

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/con_%s.pkl' % (PREFIX, role), 'wb'))
		pickle.dump(get_sizes(d), 
			open('%s/inputs/sizes/con_%s.pkl' % (PREFIX, role), 'wb'))

	# save dictionary of numpy arrays
	dump(convert_to_numpy(d), 
		'%s/inputs/%s/con_%s.gz' % (PREFIX, part, role))