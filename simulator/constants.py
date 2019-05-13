EPOCHS = 1000
MAX_TURNS = 3
N_SAMPLES = 100	# samples for gradient check
N_CAT = 6	# number of categories in categorical model
TOL_HALF = 0.02 # count concessions within this range as 1/2
ORIGIN = '2012-06-01'
MODELS = ['delay', 'con', 'round', 'nines', 'msg']
BINARY_FEATS = ['relisted', 'store', 'byr_us', 'slr_us', 'store']
COUNT_FEATS = ['byr_hist', 'slr_hist', 'fdbk_score', 'photos',
			   'slr_bos', 'slr_lstgs']

BASEDIR = '../../data/simulator/'
LDADIR = '../../data/lda/'
EXP_PATH = 'experiments.csv'
TYPES = {'epochs': 'Int64', 'mbsize': 'Int64', 'hidden': 'Int64', 'K': 'Int64',
         'layers': 'Int64', 'lr': 'Float64', 'dropout': 'Float64'}
