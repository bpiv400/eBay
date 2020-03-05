from constants import OUTPUT_DIR

# directories
LOG_DIR = OUTPUT_DIR + 'logs/'

# discriminator models
DISCRIM_MODELS = ['listings', 'threads']

# optimization parameters
NUM_WORKERS = {'first_arrival': 5,
               'init_byr': 5,
               'init_slr': 6,
               'listings': 6,
               'threads': 6}
MBSIZE = {True: 128, False: 2048}  # True for training, False for validation

# learning rate parameters
LNLR0 = [-5, -6, -7]  # initial learning rates to search over
LNLR1 = -12  # stop training when log learning rate is lower than this
LNLR_FACTOR = -1  # decrease log learning rate by this factor when training slows
FTOL = 1e-2  # decrease learning rate when relative improvement in loss is less than this

# grid search parameters
GAMMA_TOL = 0.01
GAMMA_MAX = 0.1
GAMMA_MULTIPLIER = {'first_arrival': 0,
                    'next_arrival': 0.3,
                    'hist': 0.3,
                    'delay2': 0.3,
                    'delay3': 1,
                    'delay4': 1,
                    'delay5': 1,
                    'delay6': 1,
                    'delay7': 1,
                    'con1': 0.01,
                    'con2': 1,
                    'con3': 1,
                    'con4': 1,
                    'con5': 1,
                    'con6': 1,
                    'con7': 1,
                    'msg1': 0.1,
                    'msg2': 1,
                    'msg3': 1,
                    'msg4': 1,
                    'msg5': 1,
                    'msg6': 3,
                    'init_slr': 1,
                    'init_byr': 1,
                    'listings': 0,
                    'threads': 0}
