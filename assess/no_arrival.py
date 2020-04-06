import numpy as np
import pandas as pd
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from constants import TEST, FIRST_ARRIVAL_MODEL


def main():
	# initialize dataset
	data = EBayDataset(TEST, FIRST_ARRIVAL_MODEL)

	# model predictions
	p, _ = get_model_predictions(FIRST_ARRIVAL_MODEL, data)

	# probability of no arrival
	p0 = p[:,-1]

	# histogram
	for i in np.arange(0.9, 1, 0.01):
		print('{0:1.2}% of listings have p0 > {1:.2f}'.format(
			100 * (p0 > i).mean(), i))