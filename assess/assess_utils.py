import sys, os
import numpy as np, pandas as pd, torch
from compress_pickle import load
from model.datasets.eBayDataset import eBayDataset
from model.Model import Model
from processing.processing_consts import PARAMS_PATH
from constants import INPUT_DIR, MODEL_DIR


def get_predictions(part, name):
	# create model
	sizes = load(INPUT_DIR + 'sizes/{}.pkl'.format(name))
	params = load(PARAMS_PATH)
	model = Model(name, sizes, params)
	model_path = MODEL_DIR + 'small/{}.net'.format(name)
	state_dict = torch.load(model_path)
	model.net.load_state_dict(state_dict)

	# make predictions for each example
	data = eBayDataset(part, name, sizes)
	with torch.no_grad():
		theta = model.predict_theta(data)

	return theta


def get_role_outcomes(name):
	# predictions from model
	theta = get_predictions('test_rl', name)

	# convert to distribution
	if outcome == 'msg':
		p_hat = torch.sigmoid(theta)
	else:
		p_hat = torch.exp(
			torch.nn.functional.log_softmax(theta, dim=1))

	# put in series or dataframe
	k = p_hat.size()[1]
	if k == 1:
		p_hat = pd.Series(p_hat.numpy(), index=d['index']) 
	else:
		p_hat = pd.DataFrame(p_hat.numpy(), index=d['index'], 
			columns=range(k))

	# observed outcomes
	y = pd.Series(data.d['y'], index=d['index'])

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