from compress_pickle import dump
from train.EBayDataset import EBayDataset
from assess.assess_utils import get_model_predictions
from constants import TEST, DISCRIM_MODELS


def main():
	# loop over discriminator models
	for m in DISCRIM_MODELS:
		print(m)

		# initialize dataset
		data = EBayDataset(TEST, m)
		
		# model predictions
		p, _ = get_model_predictions(m, data)

		# save predictions
		dump(p, PLOT_DIR + 'p_{}.pkl'.format(m))


if __name__ == '__main__':
	main()
