import argparse
import torch
import numpy as np
import pandas as pd
from compress_pickle import load
from processing.f_discrim.discrim_utils import get_batches, PartialDataset, get_sim_times
from processing.processing_utils import load_file, get_arrival_times, get_interarrival_period
from utils import load_model, load_featnames
from constants import VALIDATION, ENV_SIM_DIR, SIM_CHUNKS, INPUT_DIR, INDEX_DIR

NAME = 'first_con'


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
		probs = torch.exp(torch.nn.functional.log_softmax(theta, dim=1))
		a1.append(theta.cpu().numpy())
		a2.append(probs.cpu().numpy())
	return np.concatenate(a1), np.concatenate(a1)


def get_log_likelihood(y, p):
	lnL = []
	for i in range(len(y)):
		lnL.append(np.log(p[i, y.iloc[i]]))
	return np.array(lnL)



y, x = import_data()
theta, p = get_model_prediction(x)
lnL = get_log_likelihood(y, p)

