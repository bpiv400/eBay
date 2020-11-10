import numpy as np
import pandas as pd
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from constants import CLEAN_DIR
from featnames import META

# labels for meta
META_LABELS = pd.read_csv(CLEAN_DIR + 'meta.csv').set_index(META).squeeze()

# values for splitting along given variable
PRICE_CUTOFFS = [0, 5, 9, 13, 20, 25, 40, 60, 100, 225, np.inf]

# for splitting concessions
SPLITS = [0., .2, .4, .6, .8, .99]

# for finding kernel regression bandwidth
OPT = EstimatorSettings(efficient=True)

# delta for values figures
DELTA_BYR = .9
DELTA_SLR = .7

# various dimensions for plotting
POINTS = 100
VALUES_DIM = np.linspace(1 / 1000, 1, POINTS)
CON2_DIM = np.linspace(0, .8, POINTS)
NORM1_DIM = np.linspace(.4, .99, POINTS)
NORM2_DIM = np.linspace(.65, .9, POINTS)
LOG10_BIN_DIM = np.linspace(1, 3, POINTS)
LOG10_BO_DIM = np.linspace(0, 4, POINTS)
