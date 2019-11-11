import sys, pickle
from compress_pickle import load, dump
import numpy as np, pandas as pd

sys.path.append('repo/processing/4_inputs/')
from parsing_funcs import *


# loads data and calls helper functions to construct training inputs
def process_inputs(part, model, outcome):
	# initialize output dictionary and size dictionary
	d = {}

	# path name function
	getPath = lambda names: '%s/%s/%s.gz' % (PARTS_DIR, part, '_'.join(names))

	# outcome
	d['y'] = load(getPath(['y', model, outcome]))

	# fixed features
	x_lstg = cat_x_lstg(part)

    # other input variables
	x_thread = load(getPath(['x', 'thread']))
	x_offer = load(getPath(['x', 'offer']))

    # days model
	if outcome == 'days':
		d['x_fixed'] = x_lstg
		d['x_days'] = parse_time_feats_days(d)

    # other arrival interface are all feed-forward
	elif model == 'arrival':
		d['x_fixed'] = parse_fixed_feats_arrival(
			outcome, x_lstg, x_thread, x_offer)

    # byr and slr interface are all recurrent
	elif outcome == 'delay':
		d['x_fixed'] = parse_fixed_feats_delay(
		    model, x_lstg, x_thread, x_offer)

		z_start = load(getPath(['z', 'start']))
		z_role = load(getPath(['z', model]))
		d['x_time'] = parse_time_feats_delay(
		    model, d['y'].index, z_start, z_role)
	else:
		d['x_fixed'] = parse_fixed_feats_role(x_lstg, x_thread)
		d['x_time'] = parse_time_feats_role(model, outcome, x_offer)

    # dictionary of feature names
	featnames = {k: v.columns for k, v in d.items() if k.startswith('x')}

	return convert_to_arrays(d), featnames


def get_sizes(model, outcome, data):
    sizes = {}

    # fixed inputs
    sizes['fixed'] = data['x_fixed'].size()[-1]
    if outcome == 'days':
    	sizes['fixed'] += data['x_days'].size()[-1]

    # output parameters
    if outcome in ['sec', 'con']:
        sizes['out'] = 3
    elif outcome in ['days', 'hist']:
        sizes['out'] = 2
    else:
        sizes['out'] = 1

    # RNN parameters
    if model != 'arrival':
        sizes['steps'] = data['y'].size()[0]
        sizes['time'] = data['x_time'].size()[-1]

    return sizes


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser(
		description='Model of environment for bargaining AI.')
	parser.add_argument('--num', type=int, help='Model ID.')
	num = parser.parse_args().num-1
	modelid = num % len(MODEL_DIRS)
	partid = num // len(MODEL_DIRS)

	# partition, model and outcome
	part = PARTITIONS[partid]
	path = part + '/' + MODEL_DIRS[modelid]

	path = MODEL_DIRS[]
	print(path)
	model, outcome, _ = path.split('/')

	# loop over partitions
	for partition in ['train_models', 'train_rl', 'test']:
		print('\t%s' % partition)

		# input dataframes, output tensors
		tensors, featnames = process_inputs(partition, model, outcome)

		# save tensors to pickle
		dump(tensors, MODEL_DIR + path + partition + '.gz')

		# save sizes and featnames once
		if partition == 'test':
			# save featnames
			pickle.dump(featnames, 
				open(MODEL_DIR +  path + 'featnames.pkl', 'wb'))

			# get data size parameters and save
			pickle.dump(get_sizes(model, outcome, tensors), 
				open(MODEL_DIR +  path + 'sizes.pkl', 'wb'))


