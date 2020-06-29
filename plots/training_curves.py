from compress_pickle import load
import numpy as np
import pandas as pd
from plots.util import training_plot
from constants import PLOT_DIR, MODELS


def main():
	# load data
	lnl = load(PLOT_DIR + 'training_curves.pkl')

	# loop over models, create plot
	for m in MODELS:
		print(m)
		df = np.exp(pd.DataFrame.from_dict(lnl[m]))
		
		# make plot
		training_plot(m, df)


if __name__ == '__main__':
	main()
