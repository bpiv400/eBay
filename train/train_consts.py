from constants import OUTPUT_DIR

# directories
LOG_DIR = OUTPUT_DIR + 'logs/'

# optimization parameters
NUM_WORKERS = {'first_arrival': 5,
               'delay_slr': 5,
               'con_slr': 6}
MBSIZE = {True: 128, False: 2048}  # True for training, False for validation

# learning rate parameters
LNLR0 = [-5, -6, -7]  # initial learning rates to search over
LNLR1 = -12  # stop training when log learning rate is lower than this
LNLR_FACTOR = -1  # decrease log learning rate by this factor when training slows
FTOL = 1e-2  # decrease learning rate when relative improvement in loss is less than this

# grid search parameters
GRID_INC = 0.01
