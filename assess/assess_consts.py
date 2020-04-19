import numpy as np
from featnames import META, START_PRICE

# censoring threshold for threads per listing
MAX_THREADS = 4

# dimension over which to evaluate ROC curve
ROC_DIM = np.arange(0, 1 + 1e-8, 0.001)

# values for splitting along given variable
SPLIT_VALS = {META: [25, 1, 34, 7, 3, 6, 26, 27, 14, 33, 13, 24, 22],
              START_PRICE: [0, 5, 9, 13, 20, 25, 40, 60, 100, 225, np.inf]}
