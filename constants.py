from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
import torch
import numpy as np

# strings for referencing quantities related to buyer and seller models
SLR_PREFIX = 'slr'
BYR_PREFIX = 'byr'

# number of chunks for preprocessing
N_CHUNKS = 835
N_META = 35

# for partitioning
SEED = 123456
SHARES = {'train_models': 1/3, 'train_rl': 1/3}

# keep PCA components that capture this total variance
PCA_CUTOFF = 0.9

# count concessions within this range as 1/2
TOL_HALF = 0.02 

# optimization parameters
MBSIZE = 32 * 3
UPDATES = 5e6
EPOCHS = 5
LR = 1e-4

# dropout rate
DROPOUT = 0.5

# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# paths and directories
PREFIX = '/data/eBay' if torch.cuda.is_available() else 'data'
CLEAN_DIR = '%s/clean/' % PREFIX
CHUNKS_DIR = '%s/chunks/' % PREFIX
FEATS_DIR = '%s/feats/' % PREFIX
PARTS_DIR = '%s/partitions/' % PREFIX
W2V_PATH = lambda x: '%s/clean/w2v_%s.csv' %(PREFIX, x)

# partitions
PARTITIONS = ['train_models', 'train_rl', 'test']

# delete activity after lstg is open MAX_DAYS
MAX_DAYS = 30

# maximal delay times
MAX_DELAY = {
	SLR_PREFIX: 2 * 24 * 3600,
	BYR_PREFIX: 14 * 24 * 3600
}

# intervals for checking byr and offer arrivals
INTERVAL = {
	SLR_PREFIX: 30 * 60,
	BYR_PREFIX: 4 * 60 * 60
}

INTERVAL_COUNTS = {
	SLR_PREFIX: MAX_DELAY[SLR_PREFIX] / INTERVAL[SLR_PREFIX],
	BYR_PREFIX: MAX_DELAY[BYR_PREFIX] / INTERVAL[BYR_PREFIX],
	'{}_7'.format(BYR_PREFIX): MAX_DELAY[SLR_PREFIX] / INTERVAL[BYR_PREFIX]
}

# model names
MODELS = ['arrival', 'delay_byr', 'delay_slr', 'con_byr', 'con_slr']

# organizing hierarchy
LEVELS = ['slr', 'meta', 'leaf', 'product', 'cndtn', 'lstg']

# for lstg feature construction
BINARY_FEATS = ['store', 'slr_us', 'fast']
COUNT_FEATS = ['photos', 'slr_bos', 'slr_lstgs', 'fdbk_score']

# indices for byr and slr offers
IDX = {
	BYR_PREFIX: [1, 3, 5, 7],
	SLR_PREFIX: [2, 4, 6]
}

# date range and holidays
START = '2012-06-01 00:00:00'
END = '2013-05-31 23:59:59'
HOLIDAYS = Calendar().holidays(start=START, end=END)

# quantiles of accept_norm distribution
QUANTILES = [0.25, 0.5, 0.75, 1]

# timings related constants
FF_TIMINGS_LOG_DIR = 'data/logs/ff_timings/'
BATCH_TIMES = 'batch_time'
FULL_BATCH = 'full_batch'  # full batch time
DL_TIME = 'data_loader'  # time for moving to device
MOVE_TIME = 'move'  # time for moving to device
LOSS_TIME = 'loss'  # time for computing the loss
FORWARD_TIME = 'forward'  # time for forward pass of model
BACKWARD_TIME = 'backward'  # time for backproping loss
LIKELIHOOD = 'likelihood'   # log likelihood

BATCH_TIMINGS_LIST = [  # list of batch timing names
	FULL_BATCH,
	DL_TIME,
	MOVE_TIME,
	LOSS_TIME,
	FORWARD_TIME,
	BACKWARD_TIME
]

EPOCH_TIME = 'epoch_time'

# for concession model
CON_SEGMENTS = 100
CON_SEP = 1/CON_SEGMENTS
CON_DIM = torch.tensor(np.sort(np.append(np.array([0.0, 1.0]), 
	np.arange(CON_SEP/2, 1, CON_SEP)))).unsqueeze(dim=0).to(DEVICE)

MODEL_DIR = 'output fake'
