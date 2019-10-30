import sys, pickle, os
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def get_x_time(x_offer, outcome, role):
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
	excluded = ['auto', 'exp', 'reject', 'msg']
	if outcome == 'con':
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


 def get_y(x_offer, outcome, role):
 	# subset to relevant observations
 	if outcome == 'con':
		# drop zero delay and expired offers
	    mask = (x_offer.delay > 0) & ~x_offer.exp
 	elif outcome == 'msg':
 		# drop accepts and rejects
 		mask = (x_offer.con > 0) & (x_offer.con < 1)
 	s = x_offer.loc[mask, outcome]
	# subset to role
    s = s[s.index.isin(IDX[role], level='index')]
    # for concession, convert to index
    if outcome == 'con':
    	s *= 100
    	s.loc[(s > 99) & (s < 100)] = 99
    	s.loc[(s > 0) & (s < 1)] = 1
    	s = np.round(s)
    # convert to byte and unstack
    return s.astype('int8').unstack(fill_value=-1)


# loads data and calls helper functions to construct training inputs
def process_inputs(part, outcome, role):
	# path name function
	getPath = lambda names: '%s/partitions/%s/%s.gz' % \
		(PREFIX, part, '_'.join(names))

	# load dataframes
	x_lstg = load(getPath(['x', 'lstg']))
	x_thread = load(getPath(['x', 'thread']))
	x_offer = load(getPath(['x', 'offer']))
	tf = load(getPath(['tf', 'role', 'diff'])).reindex(
		index=x_offer, fill_value=0)
	x_offer = x_offer.join(tf)

	# outcome
	y = get_y(x_offer, outcome, role)

	# sort by number of turns
    turns = get_sorted_turns(y)
    y = y.reindex(index=turns.index)

	# fixed features
	x_fixed = x_lstg.reindex(index=turns.index, level='lstg').join(x_thread)

	# time features
	x_time = get_x_time(x_offer, outcome, role)

	return {'y': y.astype('int8', copy=False), 
			'turns': turns.astype('uint8', copy=False),
			'x_fixed': x_fixed.astype('float32', copy=False), 
			'x_time': x_time.astype('float32', copy=False)}


if __name__ == '__main__':
	# extract model from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1

	# partition and role
	part = PARTITIONS[num // 4]
	outcome = 'con' if num % 2 else 'msg'
	role = 'slr' if (num // 2) % 2 else 'byr'
	model = '%s_%s' % (outcome, role)
	print('%s/%s' % (part, model))

	# input dataframes, output processed dataframes
	d = process_inputs(part, outcome, role)

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/%s.pkl' % (PREFIX, model), 'wb'))

		pickle.dump(get_sizes(d), 
			open('%s/inputs/sizes/%s.pkl' % (PREFIX, model), 'wb'))

	# save dictionary of numpy arrays
	dump(convert_to_numpy(d), 
		'%s/inputs/%s/%s.gz' % (PREFIX, part, model))