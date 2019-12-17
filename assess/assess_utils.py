import sys, os
import numpy as np, pandas as pd, torch
from compress_pickle import load
from models.simulator.interface import Inputs, predict_theta
from models.simulator.model import Simulator
from constants import *


def get_role_outcomes(model):
	# create model
	sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
	simulator = Simulator(model, sizes, dropout=False)
	simulator.net.load_state_dict(
		torch.load(MODEL_DIR + '%s/0.net' % model))

	# make predictions for each example
	data = Inputs(EVAL_PART, model)
	with torch.no_grad():
		theta = predict_theta(simulator, data)

	# convert to distribution
	if outcome in ['con', 'hist']:
		p_hat = torch.exp(
			torch.nn.functional.log_softmax(theta, dim=1))
	elif outcome in ['delay', 'msg']:
		p_hat = torch.sigmoid(theta)
	elif outcome == 'arrival':
		poisson = torch.distributions.poisson.Poisson(
			torch.exp(theta))
		p_hat = torch.zeros(theta.size()[0], 12)
		for k in range(11):
			p_hat[:,k] = torch.exp(poisson.log_prob(k))
		p_hat[:,11] = 1 - torch.sum(p_hat[:,:11], dim=1)

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
	if outcome in ['con', 'msg']:
		# outcomes by role
		y_byr, p_hat_byr = get_outcomes('%s_byr' % outcome)
		y_slr, p_hat_slr = get_outcomes('%s_slr' % outcome)

		# combine
		y = pd.concat([y_byr, y_slr], dim=0).sort_index()
		p_hat = pd.concat([p_hat_byr, p_hat_slr], dim=0).sort_index()

	# placeholder for delay models
	elif outcome == 'delay':
		return None

	# no roles for arrival and hist models
	else:
		y, p_hat = get_role_outcomes(outcome)

	# sort and return
	return y.sort_index(), p_hat.sort_index()