from compress_pickle import dump
import pandas as pd
from train.EBayDataset import EBayDataset
from processing.processing_utils import input_partition, load_file
from assess.assess_utils import get_model_predictions
from constants import FIRST_ARRIVAL_MODEL, PARTS_DIR


def main():
	# partition
	part = input_partition()

	# initialize dataset
	data = EBayDataset(part, FIRST_ARRIVAL_MODEL)

	# model predictions
	p, _ = get_model_predictions(FIRST_ARRIVAL_MODEL, data)

	# probability of no arrival
	p0 = p[:,-1]

	# join with lookup
	lookup = load_file(part, 'lookup')
	s = pd.Series(p0, index=lookup.index, name='p_no_arrival')
	lookup = lookup.join(s)

	# save
	dump(lookup, PARTS_DIR + '{}/lookup.gz'.format(part))


if __name__ == '__main__':
	main()
