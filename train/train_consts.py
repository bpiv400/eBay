from constants import OUTPUT_DIR

# directories
LOG_DIR = OUTPUT_DIR + 'logs/'

# optimization parameters
NUM_WORKERS = {'arrival': 2, 'hist': 6, 'delay_byr': 4, 'delay_slr': 4,
			   'con_byr': 2, 'con_slr': 4, 'msg_byr': 3, 'msg_slr': 8}
MBSIZE = {True: 128, False: 2048}
LOGLR0 = [-2, -2.5, -3]
LOGLR1 = -5
LOGLR_INC = 0.5
FTOL = 0.99