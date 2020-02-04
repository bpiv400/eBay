from constants import OUTPUT_DIR

# directories
LOG_DIR = OUTPUT_DIR + 'logs/'

# optimization parameters
NUM_WORKERS = {'arrival': 2, 'hist': 6, 'delay_byr': 4, 'delay_slr': 4,
			   'con_byr': 2, 'con_slr': 4, 'msg_byr': 3, 'msg_slr': 8,
			   'listings': 8, 'threads': 8, 'con_byr_no7': 2}
MBSIZE = {True: 128, False: 2048}	# True for training, False for validation

# learning rate parameters
LNLR0 = [-4, -5, -6, -7]	# initial learning rates to search over
LNLR1 = -12					# stop training when log learning rate is lower than this
LNLR_FACTOR = -1			# decrease log learning rate by this factor when training slows
FTOL = 1e-2					# decrease learning rate when relative improvement in loss is less than this