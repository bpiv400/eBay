import torch
import numpy as np
import pandas as pd
from compress_pickle import load
from processing.f_discrim.discrim_utils import get_batches, PartialDataset
from utils import load_model, load_featnames
from constants import VALIDATION, INPUT_DIR, INDEX_DIR

NAME = 'delay7'


def import_data():
	d = load(INPUT_DIR + '{}/{}.gz'.format(VALIDATION, NAME))
	idx = load(INDEX_DIR + '{}/{}.gz'.format(VALIDATION, NAME))
	featnames = load_featnames(NAME)
	# reconstruct y
	y = pd.Series(d['y'], index=idx)
	# reconstruct x
	x = {}
	for k, v in d['x'].items():
		cols = featnames['offer' if 'offer' in k else k]
		x[k] = pd.DataFrame(v, index=idx, columns=cols)
	# x to numpy
	x = {k: v.values for k, v in x.items()}
	return y, x


def get_model_prediction(x):
	# create dataset
	data = PartialDataset(x)
	# create model
	net = load_model(NAME).to('cuda')
	# multinomial
	print('Generating predictions from model')
	batches = get_batches(data)
	a1, a2 = [], []
	for b in batches:
		x_b = {k: v.to('cuda') for k, v in b.items()}
		theta = net(x_b)
		lnp = torch.nn.functional.log_softmax(theta, dim=1)
		a1.append(theta.cpu().numpy())
		a2.append(lnp.cpu().numpy())
	return np.concatenate(a1), np.concatenate(a2)


def get_log_likelihood(y, lnp):
	# initialize output
	lnL = pd.Series(0.0, index=y.index)
	# arrivals
	arrival = y >= 0
	lnL[arrival] = lnp[arrival, y[arrival]]
	# non-arrivals
	cens = y < 0
	if np.sum(cens) > 0:
		y_cens = y[cens]
		p_cens = np.exp(lnp[cens, :])
		for i in range(len(y_cens)):
			lnL.loc[y_cens.index[i]] = np.log(np.sum(p_cens[i, y_cens.iloc[i]:]))
	return lnL


y, x = import_data()
theta, lnp = get_model_prediction(x)
lnL = get_log_likelihood(y, lnp)

