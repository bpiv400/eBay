import sys, pickle, os, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from processing.processing_utils import load_frames, sort_by_turns, save_files


def add_turn_indicators(df):
    '''
    Appends turn indicator variables to offer matrix
    :param df: dataframe with index ['lstg', 'thread', 'index'].
    :return: dataframe with turn indicators appended
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


def get_delay_time_feats(tf, start_time, role):
    # subset by role
    tf = tf[tf.index.isin(IDX[role], level='index')]
    # add period to tf_arrival
    tf = tf.reset_index('clock')
    tf = tf.join(start_time)
    tf['period'] = (tf.clock - tf.start_time) // INTERVAL[role]
    tf = tf.drop(['clock', 'start_time'], axis=1)
    # increment period by 1; time feats are up to t-1
    tf['period'] += 1
    # drop periods beyond censoring threshold
    tf = tf[tf.period < INTERVAL_COUNTS[role]]
    if role == 'byr':
        tf = tf[~tf.index.isin([7], level='index') | \
                (tf.period < INTERVAL_COUNTS['byr_7'])]
    # sum count features by period and return
    return tf.groupby(['lstg', 'thread', 'index', 'period']).sum()


def get_y_delay(events, role):
	# construct delay
	clock = events.clock.unstack()
	delay = pd.DataFrame(index=clock.index)
	for i in range(2, 8):
		delay[i] = clock[i] - clock[i-1]
		delay.loc[delay[i] == 0, i] = np.nan  # remove auto responses
	delay = delay.rename_axis('index', axis=1).stack().astype('int64')
	# restrict to role indices
	s = delay[delay.index.isin(IDX[role], level='index')]
	c = events.censored.reindex(index=s.index)
	# expirations
	exp = s >= MAX_DELAY[role]
	if role == 'byr':
		exp.loc[exp.index.isin([7], level='index')] = s >= MAX_DELAY['slr']
	# interval of offer arrivals and censoring
	arrival = (s[~exp & ~c] / INTERVAL[role]).astype('uint16').rename('arrival')
	cens = (s[~exp & c] / INTERVAL[role]).astype('uint16').rename('cens')
	# initialize output dataframe with arrivals
	df = arrival.to_frame().assign(count=1).set_index(
		'arrival', append=True).squeeze().unstack(
		fill_value=0).reindex(index=s.index, fill_value=0)
	# vector of censoring thresholds
	v = (arrival+1).append(cens, verify_integrity=True).reindex(
		s.index, fill_value=INTERVAL_COUNTS[role])
	if role == 'byr':
		mask = v.index.isin([7], level='index') & (v > INTERVAL_COUNTS['byr_7'])
		v.loc[mask] = INTERVAL_COUNTS['byr_7']
	# replace censored observations with -1
	for i in range(INTERVAL_COUNTS[role]):
		df[i] -= (i >= v).astype('int8')
	# sort by turns and return
	return sort_by_turns(df)


def get_y_con(x_offer, role):
	# drop zero delay and expired offers
	mask = ~x_offer.auto & ~x_offer.exp
	s = x_offer.loc[mask, outcome]
	# subset to role
	s = s[s.index.isin(IDX[role], level='index')]
	# convert to byte and return
	return s.astype('int8').sort_index()


def get_y_msg(x_offer, role):
	# drop accepts and rejects
	mask = (x_offer.con > 0) & (x_offer.con < 1)
	s = x_offer.loc[mask, outcome]
	# subset to role
	s = s[s.index.isin(IDX[role], level='index')]
	# convert to boolean and return
	return s.astype(bool).sort_index()


# loads data and calls helper functions to construct train inputs
def process_inputs(part, outcome, role):
	# function to load file
	load_file = lambda x: load('{}{}/{}.gz'.format(PARTS_DIR, part, x))

	# load dataframes
	x_offer = load_file('x_offer')
	x_thread = load_file('x_thread')

	# outcome
	if outcome == 'delay':
		events = load(CLEAN_DIR + 'offers.pkl')[['clock', 'censored']].reindex(
			index=x_offer.index, level='lstg')
		y = get_y_delay(events, role)
	elif outcome == 'con':
		y = get_y_con(x_offer, role)
	elif outcome == 'msg':
		y = get_y_msg(x_offer, role)
	else:
		raise RuntimeError('Invalid outcome: {}'.format(outcome))
	idx = y.index

	# initialize dictionary of input features
	x = load_file('x_lstg')
	x = {k: v.reindex(index=idx, level='lstg') for k, v in x.items()}

	# add thread features and turn indicators to listing features
	x_thread.loc[:, 'byr_hist'] = x_thread.byr_hist.astype('float32') / 10
	x['lstg'] = x['lstg'].join(x_thread)
	x['lstg'] = add_turn_indicators(x['lstg'])

	# dataframe of offer features for relevant threads
	threads = idx.droplevel(level='index').unique()
	df = pd.DataFrame(index=threads).join(x_offer)

	# turn features
	for i in range(1, max(IDX[role])+1):
		# offer features at turn i
		offer = df.xs(i, level='index').reindex(index=idx)
		# clean
		offer = clean_offer(offer, i, outcome, role)
		# add turn number to featname
		offer = offer.rename(lambda x: x + '_%d' % i, axis=1)
		# add turn indicators
		x['offer%d' % i] = add_turn_indicators(offer)

	if outcome in ['con', 'msg']:
		return {'y': y.astype('uint8', copy=False), 'x': x}

	# index of first x_clock for each y
	idx_clock = load_file('clock').groupby(
		['lstg', 'thread']).shift().reindex(index=idx).astype('int64')

	# normalized periods remaining at start of delay period
	lstg_start = load_file('lookup').start_time
	remaining = MAX_DAYS * 24 * 3600 - (delay_start - lstg_start)
	remaining.loc[remaining.index.isin([2, 4, 6, 7], level='index')] /= \
		MAX_DELAY['slr']
	remaining.loc[remaining.index.isin([3, 5], level='index')] /= \
		MAX_DELAY['byr']
	remaining = np.minimum(remaining, 1)

	# time features
	tf_delay = load_file('tf_delay')
	start_time = events.clock.rename('start_time').groupby(
		['lstg', 'thread']).shift().dropna().astype('int64')
	tf = get_delay_time_feats(tf, start_time, role)


	return {'y': y.astype('int8', copy=False), 'x': x,
			'idx_clock': idx_clock.astype('int64', copy=False),
			'remaining': remaining.astype('float32', copy=False),
			'tf': tf.astype('float32', copy=False)}


if __name__ == '__main__':
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--part', type=str)
	parser.add_argument('--outcome', type=str)
	parser.add_argument('--role', type=str)
	args = parser.parse_args()
	part, outcome, role = args.part, args.outcome, args.role
	name = '%s_%s' % (outcome, role)
	print('%s/%s' % (part, name))

	# input dataframes, output processed dataframes
	d = process_inputs(part, outcome, role)

	# save various output files
	save_files(d, part, name)
