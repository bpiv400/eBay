from constants import OUTPUT_DIR

# directories
LOG_DIR = OUTPUT_DIR + 'logs/'

# optimization parameters
NUM_WORKERS = 8
MBSIZE = {True: 128, False: 1024}
LOGLR0 = -3
LOGLR1 = -6
LOGLR_INC = 0.5
FTOL = 0.99