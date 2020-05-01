from constants import PREFIX, HOUR, MINUTE, MAX_DELAY, HIST_QUANTILES, \
    CON_MULTIPLIER, ARRIVAL_MODELS, BYR_HIST_MODEL, DELAY_MODELS, \
    CON_MODELS, MSG_MODELS, INIT_POLICY_MODELS, INIT_VALUE_MODELS, \
    DISCRIM_MODELS

# directories
CLEAN_DIR = '%s/clean/' % PREFIX
CHUNKS_DIR = '%s/chunks/' % PREFIX
FEATS_DIR = '%s/feats/' % PREFIX
PCTILE_DIR = '%s/pctile/' % PREFIX
W2V_DIR = '%s/w2v/' % PREFIX

# vocabulary size for embeddings
VOCAB_SIZE = 32

# create new chunk once number of listings exceeds CUTOFF
CUTOFF = 1e5

# for partitioning
SEED = 123456
SHARES = {'train_models': 0.6, 'train_rl': 0.16, 'test_rl': 0.04}

# minimum number of listings that define a category
MIN_COUNT = 1000

# number of observations in small dataset
N_SMALL = 100000

# organizing hierarchy
LEVELS = ['slr', 'cat', 'cndtn', 'lstg']

# quantiles of accept_norm distribution
QUANTILES = [0.25, 0.5, 0.75, 1]

# data types for csv read
OTYPES = {'lstg': 'int64',
          'thread': 'int64',
          'index': 'uint8',
          'clock': 'int64', 
          'price': 'float64', 
          'accept': bool,
          'reject': bool,
          'censored': bool,
          'message': bool}

TTYPES = {'lstg': 'int64',
          'thread': 'int64',
          'byr': 'int64',
          'byr_hist': 'int64',
          'bin': bool,
          'byr_us': bool}

LTYPES = {'lstg': 'int64',
          'slr': 'int64',
          'meta': 'uint8',
          'cat': str,
          'cndtn': 'uint8',
          'start_date': 'uint16',
          'end_time': 'int64',
          'fdbk_score': 'int64',
          'fdbk_pstv': 'int64',
          'start_price': 'float64',
          'photos': 'uint8',
          'slr_lstgs': 'int64',
          'slr_bos': 'int64',
          'decline_price': 'float64',
          'accept_price': 'float64',
          'store': bool,
          'slr_us': bool,
          'flag': bool,
          'fast': bool}


# features for chunks
LVARS = ['cat', 'cndtn', 'start_date', 'end_time',
         'start_price', 'start_price_pctile', 'arrival_rate', 'flag']
TVARS = ['byr_hist', 'bin']
OVARS = ['clock', 'price', 'accept', 'reject', 'censored', 'message']

# intervals for checking offer arrivals
INTERVAL = {1: HOUR,
            2: 5 * MINUTE,
            3: 30 * MINUTE,
            4: 5 * MINUTE,
            5: 30 * MINUTE,
            6: 5 * MINUTE,
            7: 5 * MINUTE}

INTERVAL_COUNTS = {i: int(MAX_DELAY[i] / INTERVAL[i]) for i in INTERVAL.keys()}

# size of model output
NUM_OUT = dict()
for m in ARRIVAL_MODELS[:-1]:
    NUM_OUT[m] = INTERVAL_COUNTS[1] + 1
NUM_OUT[BYR_HIST_MODEL] = HIST_QUANTILES
for m in DELAY_MODELS:
    turn = int(m[-1])
    NUM_OUT[m] = INTERVAL_COUNTS[turn] + 1
for m in CON_MODELS[:-1] + INIT_POLICY_MODELS:
    NUM_OUT[m] = CON_MULTIPLIER + 1
for m in [CON_MODELS[-1]] + MSG_MODELS + DISCRIM_MODELS + INIT_VALUE_MODELS:
    NUM_OUT[m] = 1

# monthly discount rate for agents
MONTHLY_DISCOUNT = 0.995
