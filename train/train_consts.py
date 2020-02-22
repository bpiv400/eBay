from constants import OUTPUT_DIR

# directories
LOG_DIR = OUTPUT_DIR + 'logs/'

# optimization parameters
NUM_WORKERS = {'first_arrival': 5,
               'next_arrival': 8,
               'hist': 8,
               'first_con': 8,
               'first_msg': 8,
               'delay_byr': 8,
               'delay_slr': 5,
               'con_byr': 8,
               'con_slr': 6,
               'msg_byr': 8,
               'msg_slr': 8,
               'listings': 8,
               'threads': 8}
MBSIZE = {True: 128, False: 2048}  # True for training, False for validation

# learning rate parameters
LNLR0 = [-5, -6, -7]  # initial learning rates to search over
LNLR1 = -12  # stop training when log learning rate is lower than this
LNLR_FACTOR = -1  # decrease log learning rate by this factor when training slows
FTOL = 1e-2  # decrease learning rate when relative improvement in loss is less than this
