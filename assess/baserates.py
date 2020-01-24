import sys, argparse
from compress_pickle import load
import numpy as np
from constants import INPUT_DIR
from featnames import TURN_FEATS


def by_turn(y, turns, f):
	lnL = 0.0
	for i in range(np.shape(turns)[-1]):
		lnL += f(y[turns[:, i]]) * turns[:, i].sum()
	# left-out turn
	other = ~np.amax(turns, axis=1)
	lnL += f(y[other]) * other.sum()
	return lnL / len(y)


def binary_lnL(y):
	p = y.mean()
	return p * np.log(p) + (1-p) * np.log(1-p)


def categorical_lnL(y):
	values = np.unique(y)
	p = np.zeros(len(values))
	for i, val in enumerate(values):
		p[i] = (y == val).mean()
	assert np.abs(p.sum() - 1) <= 2 * sys.float_info.epsilon
	return np.sum(p * np.log(p))


def main():
	# extract parameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--part', type=str)
	parser.add_argument('--name', type=str)
	args = parser.parse_args()
	part, name = args.part, args.name
	print('{}/{}'.format(part, name))

	# function
	if 'msg' in name:
		f = binary_lnL
	elif 'con' in name or (name == 'hist'):
		f = categorical_lnL
	else:
		raise NotImplementedError()	# arrival and delay outcomes

	# load data
	d = load(INPUT_DIR + '{}/{}.gz'.format(part, name))
	y = d['y']

	if 'delay' in name or 'con' in name or 'msg' in name:
		featnames = load(INPUT_DIR + 'featnames/{}.pkl'.format(name))['offer']
		idx = [featnames.index(k) for k in TURN_FEATS[name]]
		turns = d['x']['offer1'][:, idx].astype(bool)

	# simple baserate
	print('Simple baserate: {0:1.4f}'.format(f(y)))

	# by turn
	if 'delay' in name or 'con' in name or 'msg' in name:
		print('Turn-specific baserates: {0:1.4f}'.format(by_turn(y, turns, f)))


if __name__ == '__main__':
    main()