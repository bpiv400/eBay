import sys, os, re
import numpy as np, pandas as pd
from compress_pickle import load, dump
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from assess.assess_utils import *
from constants import *

def get_lnL(clf, data):
	lnp_hat = clf.predict_log_proba(data['x'])
	labels = clf.classes_
	lnL = 0.0
	for i in range(len(labels)):
		lnL += (data['y'] == labels[i]) * lnp_hat[:,i]
	return np.mean(lnL)


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    model = args.model

    # load training and validation data
	train = load(INPUT_DIR + 'train_models/%s.gz' % model)
	test = load(INPUT_DIR + 'train_rl/%s.gz' % model)

	# drop turn indicators from offer components
	if 'offer1' in train['x']:
		featnames = load(INPUT_DIR + 'featnames/%s.pkl' % model)
		for i in range(1, 8):
			key = 'offer%d' % i
			if key in train['x']:
				# columns number of turn indicators
				a = featnames['x'][key]
				cols = [i for i in range(len(a)) \
					if a[i].startswith('t') and len(a[i]) == 2]
				# drop columns from training and test
				train['x'][key] = np.delete(train['x'][key], cols, axis=1)
				test['x'][key] = np.delete(test['x'][key], cols, axis=1)

	# collapse x into single numpy array
	assert train['x'].keys() == test['x'].keys()
	train['x'] = np.hstack(list(train['x'].values()))	
	test['x'] = np.hstack(list(test['x'].values()))

	# logistic regression
	optimizer = LogisticRegression(solver='saga', verbose=1, C=np.inf)
	clf = optimizer.fit(train['x'], train['y'])