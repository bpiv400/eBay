from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar

# count concessions within this range as 1/2
TOL_HALF = 0.02 

# optimization parameters
MBSIZE = 32 * 3
DROPOUT = 0.5
UPDATES = 5e6
LR = 1e-4

# paths and directories
DATA_PATH = 'data/partitions/'
CHUNKS_DIR = 'data/chunks/'
LDA_DIR = 'data/lda/'
MODEL_DIR = 'models/'
EXP_PATH = 'repo/simulator/experiments/'

# maximal delay times
MAX_DELAY = {'slr': 2 * 24 * 3600, 'byr': 14 * 24 * 3600}

# censoring threshold for days model
MAX_DAYS = 30

# intervals for checking byr and offer arrivals
INTERVAL = {'slr': 15 * 60, 'byr': 3 * 60 * 60}

# organizing hierarchy
LEVELS = ['slr', 'meta', 'leaf', 'title', 'cndtn', 'lstg']

# for lstg feature construction
META_OTHER = [0, 19]
BINARY_FEATS = ['relisted', 'store', 'slr_us', 'store', 'fast']
COUNT_FEATS = ['photos', 'slr_bos', 'slr_lstgs', 'fdbk_score']

# sequence of arrival models
ARRIVAL_MODELS = ['days', 'loc', 'hist', 'bin', 'sec', 'msg', 'con', 'round', 'nines']

# indices for byr and slr offers
IDX = {'byr': [1, 3, 5, 7], 'slr': [2, 4, 6]}

# date range and holidays
START = '2012-06-01 00:00:00'
END = '2013-05-31 23:59:59'
HOLIDAYS = Calendar().holidays(start=START, end=END)

# count feats for hierarchical time features
CT_FEATS = ['lstg', 'thread', 'slr_offer', 'byr_offer', 'accept']

# quantiles of accept_norm distribution
QUANTILES = [0.25, 0.5, 0.75, 1]