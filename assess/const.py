import numpy as np
import pandas as pd
from constants import CLEAN_DIR
from featnames import META

# censoring threshold for threads per listing
MAX_THREADS = 4

# dimension over which to evaluate ROC curve
ROC_DIM = np.arange(0, 1 + 1e-8, 0.001)

# labels for meta
META_LABELS = pd.read_csv(CLEAN_DIR + 'meta.csv').set_index(META).squeeze()

# values for splitting along given variable
PRICE_CUTOFFS = [0, 5, 9, 13, 20, 25, 40, 60, 100, 225, np.inf]
