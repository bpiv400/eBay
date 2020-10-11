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

OPT = EstimatorSettings(efficient=True)
