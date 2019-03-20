MAX_TURNS = 3
N_SAMPLES = 100	# samples for gradient check
TOL_HALF = 0.02 # count concessions within this range as 1/2
MODELS = ['delay', 'con', 'round', 'nines', 'msg']
BINARY_FEATS = ['relisted', 'store', 'byr_us', 'slr_us']
COUNT_FEATS = ['byr_hist', 'slr_hist', 'fdbk_score', 'photos', 'views',
			   'wtchrs', 'store', 'slr_bos', 'slr_lstgs']
BASEDIR = '../../data/simulator/'
EXP_PATH = 'experiments.csv'
TYPES = {'epochs': 'Int64', 'mbsize': 'Int64', 'hidden': 'Int64', 'K': 'Int64',
         'layers': 'Int64', 'lr': 'Float64', 'dropout': 'Float64'}
