import numpy as np, pandas as pd
import torch, argparse, sys
sys.path.append('repo/')
from constants import *
sys.path.append('repo/processing/')
from parsing_funcs import *


# loads data and calls helper functions to construct training inputs
def process_inputs(source, model, outcome):
	# initialize output dictionary and size dictionary
	d = {}

	# path name function
	getPath = lambda names: DATA_PATH + source + '/' + '_'.join(names) + '.pkl'

	# outcome
	d['y'] = pickle.load(open(getPath(['y', model, outcome]), 'rb'))

    # load dataframes of input variables
	x_lstg = pickle.load(open(getPath(['x', 'lstg']), 'rb'))
	x_thread = pickle.load(open(getPath(['x', 'thread']), 'rb'))
	x_offer = pickle.load(open(getPath(['x', 'offer']), 'rb'))

    # days model is recurrent
	if outcome == 'days':
		d['x_fixed'] = x_lstg
		d['x_time'] = parse_time_feats_days(d)

    # other arrival models are all feed-forward
	elif model == 'arrival':
		d['x_fixed'] = parse_fixed_feats_arrival(
			outcome, x_lstg, x_thread, x_offer)

    # byr and slr models are all recurrent
	elif outcome == 'delay':
		d['x_fixed'] = parse_fixed_feats_delay(
		    model, x_lstg, x_thread, x_offer)

		z_start = pickle.load(open(getPath(['z', 'start']), 'rb'))
		z_role = pickle.load(open(getPath(['z', model]), 'rb'))
		d['x_time'] = parse_time_feats_delay(
		    model, d['y'].index, z_start, z_role)
	else:
		d['x_fixed'] = parse_fixed_feats_role(x_lstg, x_thread)
		d['x_time'] = parse_time_feats_role(model, outcome, x_offer)

    # dictionary of feature names
	featnames = {k: v.columns for k, v in d.items() if k.startswith('x')}

	return convert_to_tensors(d), featnames


def get_sizes(model, outcome, data):
    sizes = {}

    # fixed inputs
    sizes['fixed'] = data['x_fixed'].size()[-1]

    # output parameters
    if outcome in ['sec', 'con']:
        sizes['out'] = 3
    elif outcome in ['days', 'hist']:
        sizes['out'] = 2
    else:
        sizes['out'] = 1

    # RNN parameters
    if (model != 'arrival') or (outcome == 'days'):
        sizes['steps'] = data['y'].size()[0]
        sizes['time'] = data['x_time'].size()[-1]

    return sizes


if __name__ == '__main__':
	# extract model and outcome from int
	parser = argparse.ArgumentParser(
		description='Model of environment for bargaining AI.')
	parser.add_argument('--id', type=int, help='Model ID.')
	path = MODEL_DIRS[parser.parse_args().id-1]
	print(path)
	model, outcome, _ = path.split('/')

	# loop over partitions
	for partition in ['train_models', 'train_rl', 'test']:
		print('\t%s' % partition)

		# input dataframes, output tensors
		tensors, featnames = process_inputs(partition, model, outcome)

		# save tensors to pickle
		pickle.dump(tensors, 
			open(MODEL_DIR + path + partition + '.pkl', 'wb'), protocol=4)

		# save sizes and featnames once
		if partition == 'test':
			# save featnames
			pickle.dump(featnames, 
				open(MODEL_DIR +  path + 'featnames.pkl', 'wb'))

			# get data size parameters and save
			pickle.dump(get_sizes(model, outcome, tensors), 
				open(MODEL_DIR +  path + 'sizes.pkl', 'wb'))


