from constants import OUTPUT_DIR

# directories
LOG_DIR = '%slogs/' % OUTPUT_DIR
EXPS_DIR = '%sexps/' % OUTPUT_DIR

# optimization parameters
NUM_WORKERS = 0
MBSIZE = {True: 128, False: 2000}
LOGLR0 = -3
LOGLR1 = -6
LOGLR_INC = 0.5
FTOL = 0.995

# neural net parameters
LAYERS_EMBEDDING = 4
LAYERS_FULL = 8
HIDDEN = 1024