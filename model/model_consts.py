from constants import OUTPUT_DIR

# directories
LOG_DIR = OUTPUT_DIR + 'logs/'
MODEL_DIR = OUTPUT_DIR + 'models/'

# optimization parameters
NUM_WORKERS_FF = 0
NUM_WORKERS_RNN = 2
MBSIZE = {True: 128, False: 2000}
LOGLR0 = -3
LOGLR1 = -6
LOGLR_INC = 0.5
FTOL = 0.995

# neural net parameters
LAYERS_EMBEDDING = 4
LAYERS_FULL = 8
HIDDEN = 1024
LSTM_HIDDEN = 1024