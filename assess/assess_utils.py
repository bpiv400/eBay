import sys, os
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from compress_pickle import load
from model.datasets.eBayDataset import eBayDataset
from model.Model import Model
from constants import INPUT_DIR, MODEL_DIR, INDEX_DIR, PARAMS_PATH


def get_role_outcomes(part, name):
	# create model
	sizes = load(INPUT_DIR + 'sizes/{}.pkl'.format(name))
	params = load(PARAMS_PATH)
	model = Model(name, sizes, params)
	model_path = MODEL_DIR + '{}.net'.format(name)
	state_dict = torch.load(model_path)
	model.net.load_state_dict(state_dict)

	# make predictions for each example
	data = eBayDataset(part, name)
	with torch.no_grad():
		theta = model.predict_theta(data)

	# pandas index
	idx = load(INDEX_DIR + '{}/{}.gz'.format(part, name))

	# convert to distribution
	if outcome == 'msg':
		p_hat = torch.sigmoid(theta)
		p_hat = pd.Series(p_hat.numpy(), index=idx)
	else:
		p_hat = torch.exp(F.log_softmax(theta, dim=-1))
		p_hat = pd.DataFrame(p_hat.numpy(), 
			index=idx, columns=range(p_hat.size()[1]))

	# observed outcomes
	y = pd.Series(data.d['y'], index=idx)

	return y, p_hat


def get_outcomes(outcome):
	if outcome in ['delay', 'con', 'msg']:
		# outcomes by role
		y_byr, p_hat_byr = get_outcomes('%s_byr' % outcome)
		y_slr, p_hat_slr = get_outcomes('%s_slr' % outcome)

		# combine
		y = pd.concat([y_byr, y_slr], dim=0)
		p_hat = pd.concat([p_hat_byr, p_hat_slr], dim=0)

	# no roles for arrival and hist models
	else:
		y, p_hat = get_role_outcomes(outcome)

	# sort and return
	return y.sort_index(), p_hat.sort_index()