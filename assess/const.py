import numpy as np
import pandas as pd
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from agent.const import DELTA_SLR
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
DELTA_ASSESS = .7

# agent names
SLR_NAMES = {DELTA_SLR[0]: 'Impatient agent',
             DELTA_SLR[1]: 'Patient agent'}

# various dimensions for plotting
POINTS = 100
VALUES_DIM = np.linspace(1 / 1000, 1, POINTS)
CON2_DIM = np.linspace(0, .8, POINTS)
NORM1_DIM = np.linspace(.4, .9, POINTS)
NORM1_DIM_LONG = np.linspace(.4, 1., POINTS)
NORM2_DIM = np.linspace(.65, .9, POINTS)
NORM2_DIM_LONG = np.linspace(.6, 1, POINTS)
NORM3_DIM = np.linspace(.5, .9, POINTS)
NORM5_DIM = np.linspace(.55, .9, POINTS)
LOG10_BIN_DIM = np.linspace(1, 3, POINTS)
LOG10_BO_DIM = np.linspace(0, 4, POINTS)
BYR_NORM_DIMS = {1: NORM1_DIM, 3: NORM3_DIM, 5: NORM5_DIM}

# for 2D plotting
NORM1_DIM_SHORT = np.linspace(.4, .85, 50)

xx1, xx2 = np.meshgrid(NORM1_DIM_SHORT, np.linspace(1, 2.5, 50))
NORM1_BIN_MESH = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)
