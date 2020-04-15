from compress_pickle import load
import numpy as np
import pandas as pd
from plots.plots_utils import training_plot
from constants import PLOT_DIR, MODELS


def main():
	# load data
	lnl = load(PLOT_DIR + 'training_curves.pkl')

	# loop over models, create plot
	for m in MODELS:
		print(m)
		df = pd.DataFrame.from_dict(np.exp(lnl[m]))
		
		# make plot
		training_plot('training_{}'.format(m), df)


if __name__ == '__main__':
	main()
