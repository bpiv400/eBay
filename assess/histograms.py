import sys, os, argparse
import numpy as np, pandas as pd, torch
import torch.nn.functional as funcs
from compress_pickle import load, dump
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.simulator.interface import Inputs, predict_theta
from models.simulator.model import Simulator
from constants import *

PART = 'train_rl'

if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    model = args.model

	# create model
	sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
	simulator = Simulator(model, sizes, dropout=False)
	simulator.net.load_state_dict(
		torch.load(MODEL_DIR + '%s/0.net' % model))

	# make predictions for each example
	data = Inputs(PART, model)
	with torch.no_grad():
		theta = predict_theta(simulator, data).to('cpu')

	# convert to distribution
	if 'con' in model:
		p_hat = torch.exp(funcs.log_softmax(theta, dim=1)).numpy()

	# lookup file
	lookup = load(PARTS_DIR + '%s/lookup.gz' % PART)

	# overall histograms
	bins = np.arange(101)
	p0 = np.array([np.mean(data.d['y'] == b) for b in bins])
	p1 = np.mean(p_hat, axis=0)

	plt.bar(bins, p0, alpha=0.5, label='data')
	plt.bar(bins, p1, alpha=0.5, label='model')
	plt.legend(loc='upper right')
	plt.show()