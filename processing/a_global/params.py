from compress_pickle import dump
from constants import PARAMS_PATH

# neural net parameters
PARAMS = {'layers_embedding': 4,
		  'layers_full': 8,
		  'hidden': 1024,
		  'batchnorm': True,
		  'dropout': False,
		  'affine': True}

# save dictionary of neural net parameters
dump(PARAMS, PARAMS_PATH)