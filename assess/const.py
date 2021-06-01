import numpy as np
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from agent.const import DELTA_SLR

# labels for meta
META_LABELS = {1: 'Collectibles',
               99: 'Everything Else',
               220: 'Toys',
               237: 'Dolls',
               260: 'Stamps',
               267: 'Books',
               281: 'Jewelry',
               293: 'Consumer Electronics',
               550: 'Art',
               619: 'Musical Instruments',
               625: 'Cameras',
               870: 'Pottery',
               888: 'Sporting Goods',
               1249: 'Video Games',
               1281: 'Pet Supplies',
               1305: 'Tickets',
               2984: 'Baby',
               3252: 'Travel',
               11116: 'Coins',
               11232: 'DVDs',
               11233: 'Music',
               11450: 'Clothing',
               11700: 'Home',
               12576: 'Business',
               14339: 'Crafts',
               15032: 'Cell Phones',
               20081: 'Antiques',
               26395: 'Health & Beauty',
               45100: 'Entertainment Memorabilia',
               58058: 'Computers',
               64482: 'Sports Memorabilia'}

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
