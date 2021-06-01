import numpy as np
import pandas as pd
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from agent.const import DELTA_SLR
from constants import CLEAN_DIR
from featnames import META

# labels for meta
META_LABELS = pd.read_csv(CLEAN_DIR + 'meta.csv').set_index(META).squeeze()

# for finding kernel regression bandwidth
OPT = EstimatorSettings(efficient=True)

# agent names
SLR_NAMES = {DELTA_SLR[0]: 'Impatient agent',
             DELTA_SLR[1]: 'Patient agent'}

# various dimensions for plotting
POINTS = 100
VALUES_DIM = np.linspace(1 / 1000, 1, POINTS)
NORM1_DIM = np.linspace(.4, .9, POINTS)
NORM1_DIM_LONG = np.linspace(.5, 1., POINTS)
NORM1_DIM_SHORT = np.linspace(.4, .85, 50)
LOG10_BIN_DIM = np.linspace(1, 2.75, POINTS)
LOG10_BIN_DIM_SHORT = np.linspace(1, 2.5, 50)
LOG10_BO_DIM = np.linspace(0, 4, POINTS)

# for 2D plotting
xx1, xx2 = np.meshgrid(NORM1_DIM_SHORT, LOG10_BIN_DIM_SHORT)
NORM1_BIN_MESH = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)

xx1, xx2 = np.meshgrid(np.linspace(.65, .95, 50), LOG10_BIN_DIM_SHORT)
NORM2_BIN_MESH = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)
