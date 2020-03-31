from constants import OUTPUT_DIR

# directories
LOG_DIR = OUTPUT_DIR + 'logs/'

# optimization parameters
NUM_WORKERS = {'first_arrival': 4,
               'hist': 7,
               'con1': 7,
               'init_byr': 5,
               'init_slr': 6,
               'listings': 6,
               'threads': 6,
               'threads_no_tf': 7}
MBSIZE = {True: 128, False: 2048}  # True for training, False for validation

# learning rate parameters
LNLR0 = [-5, -6, -7]  # initial learning rates to search over
LNLR1 = -12  # stop training when log learning rate is lower than this
LNLR_FACTOR = -1  # decrease log learning rate by this factor when training slows
FTOL = 1e-2  # decrease learning rate when relative improvement in loss is less than this

# dropout grid
INT_DROPOUT = 10
MAX_DROPOUT = 7
DROPOUT_GRID = []
for i in range(0, MAX_DROPOUT):
    for j in range(i + 1, MAX_DROPOUT + 1):
        DROPOUT_GRID.append([i, j])

# type of normalization
NORM_TYPE = {'init_slr': 'weight',
			 'init_byr': 'weight'}
