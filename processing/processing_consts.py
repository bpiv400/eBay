from constants import PREFIX, HOUR, MINUTE, MAX_DELAY, HIST_QUANTILES, CON_MULTIPLIER

# directories
CLEAN_DIR = '%s/clean/' % PREFIX
CHUNKS_DIR = '%s/chunks/' % PREFIX
FEATS_DIR = '%s/feats/' % PREFIX
PCTILE_DIR = '%s/pctile/' % PREFIX
W2V_DIR = '%s/w2v/' % PREFIX
LOG_DIR = '%s/outputs/logs/' % PREFIX

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
NUM_OUT = {'first_arrival': INTERVAL_COUNTS[1] + 1,
           'next_arrival':  INTERVAL_COUNTS[1] + 1,
           'hist': HIST_QUANTILES,
           'delay2': INTERVAL_COUNTS[2] + 1,
           'delay3': INTERVAL_COUNTS[3] + 1,
           'delay4': INTERVAL_COUNTS[4] + 1,
           'delay5': INTERVAL_COUNTS[5] + 1,
           'delay6': INTERVAL_COUNTS[6] + 1,
           'delay7': INTERVAL_COUNTS[7] + 1,
           'con1': CON_MULTIPLIER + 1,
           'con2': CON_MULTIPLIER + 1,
           'con3': CON_MULTIPLIER + 1,
           'con4': CON_MULTIPLIER + 1,
           'con5': CON_MULTIPLIER + 1,
           'con6': CON_MULTIPLIER + 1,
           'con7': 1,
           'msg1': 1,
           'msg2': 1,
           'msg3': 1,
           'msg4': 1,
           'msg5': 1,
           'msg6': 1}
