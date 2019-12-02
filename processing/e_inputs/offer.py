import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


# appends turn indicator variables to offer matrix
def add_turn_indicators(df):
    '''
    df: dataframe with index ['lstg', 'thread', 'index'].
    '''
    indices = np.unique(df.index.get_level_values('index'))
    for i in range(len(indices)-1):
        ind = indices[i]
        featname = 't%d' % ((ind+1) // 2)
        df[featname] = df.index.isin([ind], level='index')
    return df

# deletes irrelevant feats and sets unseen feats to 0
def clean_offer(offer, i, outcome, role):
	# if turn 1, drop days and delay
	if i == 1:
		offer = offer.drop(['days', 'delay'], axis=1)
	# set features to 0 if i exceeds index
	else:
		future = i > offer.index.get_level_values(level='index')
		offer.loc[future, offer.dtypes == 'bool'] = False
		offer.loc[future, offer.dtypes != 'bool'] = 0
	# for current turn, set feats to 0
	curr = i == offer.index.get_level_values(level='index')
	if outcome == 'delay':
		offer.loc[curr, offer.dtypes == 'bool'] = False
		offer.loc[curr, offer.dtypes != 'bool'] = 0
	else:
		offer.loc[curr, 'msg'] = False
		if outcome == 'con':
			offer.loc[curr, ['con', 'norm']] = 0
			offer.loc[curr, ['split', 'auto', 'exp', 'reject']] = False
	# if buyer turn or last turn, drop auto, exp, reject
	if (i in IDX['byr']) or (i == max(IDX[role])):
		offer = offer.drop(['auto', 'exp', 'reject'], axis=1)
	# on last turn, drop msg (and concession features)
	if i == max(IDX[role]):
		offer = offer.drop('msg', axis=1)
		if outcome == 'con':
			offer = offer.drop(['con', 'norm', 'split'], axis=1)
	return offer


# concatenates offer features across turns
def get_offer_vec(x, featnames):
	feats = []
	# loop over features and turns
	for name in featnames:
		for i in range(1, max(IDX[role])+1):
			key = 'offer%d' % i
			if name in x[key]:
				newname = '%s_%d' % (name, i)
				feats.append(x[key][name].rename(newname))
	# concatenate and add turn indicators
	return add_turn_indicators(pd.concat(feats, axis=1))


# returns either a series of msg indicators or concession indices
def get_y(x_offer, outcome, role):
	# subset to relevant observations
	if outcome == 'con':
		# drop zero delay and expired offers
		mask = ~x_offer.auto & ~x_offer.exp
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
	# convert to byte and return
	return s.astype('int8').sort_index()


# loads data and calls helper functions to construct training inputs
def process_inputs(part, outcome, role):
	# path name function
	getPath = lambda names: '%s/partitions/%s/%s.gz' % \
		(PREFIX, part, '_'.join(names))

	# load dataframes
	x_offer = load(getPath(['x', 'offer']))

	# outcome
	if outcome == 'delay':
		y = load(getPath(['y', 'delay', role]))
	else:
		y = get_y(x_offer, outcome, role)
	idx = y.index

	# initialize dictionary of input features
	x = init_x(getPath, idx)

	# add thread features and turn indicators to listing features
	x_thread = load(getPath(['x', 'thread']))
	x['lstg'] = x['lstg'].join(x_thread.months_since_lstg)
	x['lstg'] = x['lstg'].join(x_thread.byr_hist.astype('float32') / 10)
	x['lstg'] = add_turn_indicators(x['lstg'])

	# save price features for later
	price_feats = x['lstg'][['start_price_pctile', 'auto_decline', 'auto_accept']]

	# dataframe of offer features for relevant threads
	threads = idx.droplevel(level='index').unique()
	df = pd.DataFrame(index=threads).join(x_offer)

	# turn features
	for i in range(1, max(IDX[role])+1):
		# offer features at turn i
		offer = df.xs(i, level='index').reindex(index=idx)
		# clean
		offer = clean_offer(offer, i, outcome, role)
		# add turn indicators
		offer = add_turn_indicators(offer)
		# append with price features
		x['offer%d' % i] = pd.concat([price_feats, offer], axis=1)

	# offer type features with turn indicators
	for k, v in OFFER_GROUPS.items():
		x[k] = get_offer_vec(x, v)

	# add price feats to x['price']
	x['price'] = pd.concat([price_feats, x['price']], axis=1)

	# if not delay model, return
	if outcome in ['con', 'msg']:
		return {'y': y.astype('uint8', copy=False), 'x': x}

	# clock features by minute
	x_clock = create_x_clock()

	# index of first x_clock for each y
	delay_start = load(getPath(['clock'])).groupby(
		['lstg', 'thread']).shift().reindex(index=idx).astype('int64')
	idx_clock = delay_start // 60

	# normalized periods remaining at start of delay period
	lstg_start = load(getPath(['lookup'])).start_date.astype(
		'int64') * 24 * 3600
	remaining = MAX_DAYS * 24 * 3600 - (delay_start - lstg_start)
	remaining.loc[remaining.index.isin([2, 4, 6, 7], level='index')] /= \
		MAX_DELAY['slr']
	remaining.loc[remaining.index.isin([3, 5], level='index')] /= \
		MAX_DELAY['byr']
	remaining = np.minimum(remaining, 1)

	# time features
	tf = load(getPath(['tf', 'delay', 'diff', role]))

	return {'y': y.astype('int8', copy=False), 'x': x,
			'x_clock': x_clock.astype('float32', copy=False),
			'idx_clock': idx_clock.astype('int64', copy=False),
			'remaining': remaining.astype('float32', copy=False),
			'tf': tf.astype('float32', copy=False)}


if __name__ == '__main__':
	# extract model from int
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', type=int)
	num = parser.parse_args().num-1

	# partition and role
	part = PARTITIONS[num // 6]
	outcome = ['delay', 'con', 'msg'][num % 3]
	role = 'slr' if (num // 3) % 2 else 'byr'
	model = '%s_%s' % (outcome, role)
	print('%s/%s' % (part, model))

	# input dataframes, output processed dataframes
	d = process_inputs(part, outcome, role)

	# save featnames and sizes
	if part == 'train_models':
		pickle.dump(get_featnames(d), 
			open('%s/inputs/featnames/%s.pkl' % (PREFIX, model), 'wb'))

		pickle.dump(get_sizes(d, model), 
			open('%s/inputs/sizes/%s.pkl' % (PREFIX, model), 'wb'))

	# create dictionary of numpy arrays
	d = convert_to_numpy(d)

	# save as dataset
	dump(d, '%s/inputs/%s/%s.gz' % (PREFIX, part, model))

	# save small dataset
	if part == 'train_models':
		dump(create_small(d), '%s/inputs/small/%s.gz' % (PREFIX, model))
