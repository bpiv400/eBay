from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar

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
LR = 1e-4

# dropout rate
DROPOUT = 0.5

# paths and directories
CLEAN_DIR = 'data/clean/'
DATA_PATH = 'data/partitions/'
CHUNKS_DIR = 'data/chunks/'
FEATS_DIR = 'data/feats/'
PCA_DIR = 'data/pca/'
PARTS_DIR = 'data/partitions/'
MODEL_DIR = 'models/'
EXP_PATH = 'repo/simulator/experiments/'
W2V_PATH = lambda x: 'data/clean/w2v_' + x + '.csv'

# model directories
MODEL_DIRS = ['arrival/days/',
		  	  'arrival/bin/',
		  	  'arrival/loc/',
		  	  'arrival/hist/',
		  	  'arrival/sec/',
		  	  'byr/delay/',
		  	  'byr/accept/',
		  	  'byr/reject/',
		  	  'byr/con/',
		  	  'byr/msg/',
		  	  'byr/round/',
		  	  'byr/nines/',
		  	  'slr/delay/',
		  	  'slr/accept/',
		  	  'slr/reject/',
		  	  'slr/con/',
		  	  'slr/msg/',
		  	  'slr/round/',
		  	  'slr/nines/']

# maximal delay times
MAX_DELAY = {'slr': 2 * 24 * 3600, 'byr': 14 * 24 * 3600}

# censoring threshold for days model
MAX_DAYS = 30

# intervals for checking byr and offer arrivals
INTERVAL = {'slr': 15 * 60, 'byr': 3 * 60 * 60}

INTERVAL_COUNTS = {
					'slr': MAX_DELAY['slr'] / INTERVAL['slr'],
					'byr': MAX_DELAY['byr'] / INTERVAL['byr'],
					'byr_7': MAX_DELAY['slr'] / INTERVAL['byr']
				}

# organizing hierarchy
LEVELS = ['slr', 'meta', 'leaf', 'product', 'title', 'cndtn', 'lstg']

# for lstg feature construction
BINARY_FEATS = ['store', 'slr_us', 'fast']
COUNT_FEATS = ['photos', 'slr_bos', 'slr_lstgs', 'fdbk_score']

# sequence of arrival models
ARRIVAL_MODELS = ['days', 'loc', 'hist', 'bin', 'sec', 'msg', 'con', 'round', 'nines']

# indices for byr and slr offers
IDX = {'byr': [1, 3, 5, 7], 'slr': [2, 4, 6]}

# date range and holidays
START = '2012-06-01 00:00:00'
END = '2013-05-31 23:59:59'
HOLIDAYS = Calendar().holidays(start=START, end=END)

# quantiles of accept_norm distribution
QUANTILES = [0.25, 0.5, 0.75, 1]