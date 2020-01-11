from compress_pickle import dump
from constants import PARAMS_PATH

# neural net parameters
PARAMS = {'layers_embedding': 4,
		  'layers_full': 8,
		  'hidden': 1024,
		  'batchnorm': True,
		  'dropout': False,
		  'layernorm': False,
		  'dropout_lstm': 0.5,
		  'affine': True,
		  'hidden_lstm': 256}

# save dictionary of neural net parameters
dump(PARAMS, PARAMS_PATH)