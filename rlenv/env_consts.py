from utils import unpickle
import torch

# time feats
SLR_OFFERS = 'slr_offers'
BYR_OFFERS = 'byr_offers'
SLR_OFFERS_OPEN = 'slr_offers_open'
BYR_OFFERS_OPEN = 'byr_offers_open'
SLR_BEST = 'slr_best'
BYR_BEST = 'byr_best'
SLR_BEST_OPEN = 'slr_best_open'
BYR_BEST_OPEN = 'byr_best_open'

TIME_FEATS = [
    SLR_OFFERS,
    BYR_OFFERS,
    SLR_OFFERS_OPEN,
    BYR_OFFERS_OPEN,
    SLR_BEST,
    BYR_BEST,
    SLR_BEST_OPEN,
    BYR_BEST_OPEN
]

# outcomes
SLR_OUTCOMES = ['delay', 'con', 'norm', 'split', 'round', 'nines', 'msg', 'auto', 'reject', 'exp']
BYR_OUTCOMES = ['delay', 'con', 'norm', 'split', 'round', 'nines', 'msg']

# turn indices
SLR_TURN_INDS = ['t1', 't2']
BYR_TURN_INDS = SLR_TURN_INDS + ['t3']


# filenames
DATA_DIR = 'data/partitions/train_rl'
MODEL_DIR = 'models'
COMPOSER_DIR = '{}/composer'.format(MODEL_DIR)
LSTG_FILENAME = '{}/lstg.h5df'.format(DATA_DIR)
FIXED_COLS_FILENAME = '{}/x_lstg_cols.pkl'.format(DATA_DIR)
EXPERIMENT_PATH = 'repo/rlenv/experiments.csv'
FEATNAMES_FILENAME = 'featnames.pkl'
SIZES_FILENAME = 'sizes.pkl'
LSTG_COLS = unpickle(FIXED_COLS_FILENAME)

# clock feats
DAYS_CLOCK_FEATS = ['days', 'holiday', 'dow0', 'dow1', 'dow2', 'dow3', 'dow4', 'dow5']
FF_CLOCK_FEATS = ['focal_{}'.format(feat) for feat in DAYS_CLOCK_FEATS]
OFFER_CLOCK_FEATS = DAYS_CLOCK_FEATS + ['minutes']
# remove when Etan adds days to delay model cock
DELAY_CLOCK_FEATS = [feat for feat in OFFER_CLOCK_FEATS if feat != 'days']


# temporal constants
MONTH = 30 * 24 * 3600
DAY = 24 * 3600
EXPIRATION = 48 * 60 * 60

# map names
LSTG_MAP = 'lstg'
CLOCK_MAP = 'clock'
BYR_ATTR_MAP = 'byr_attr'
BYR_US_MAP = 'byr_us'
BYR_HIST_MAP = 'byr_hist'
OUTCOMES_MAP = 'outcomes'
SIZE = 'size'
PERIODS_MAP = 'periods'
TIME_MAP = 'time'
TURN_IND_MAP = 'turn_inds'
DIFFS_MAP = 'diffs'
# other turn maps
O_CLOCK_MAP = 'other_clock'
O_OUTCOMES_MAP = 'other_outcomes'
O_TIME_MAP = 'other_time'
O_DIFFS_MAP = 'other_diffs'
# last turn maps
L_CLOCK_MAP = 'last_clock'
L_OUTCOMES_MAP = 'last_outcomes'
L_TIME_MAP = 'last_time'

# start map
START_CLOCK_MAP = [LSTG_COLS[clock_feat] for
                   clock_feat in DAYS_CLOCK_FEATS if clock_feat != 'days']

# zero vectors
ZERO_SLR_OUTCOMES = torch.zeros(len(SLR_OUTCOMES))
ZERO_SLR_OUTCOMES[[7, 8]] = 1


# remove when Etan adds days to delay model cock
DELAY_CLOCK_ZEROS = torch.zeros(len(DELAY_CLOCK_FEATS))



BYR_ATTRS = ['byr_us', 'byr_hist']
FIXED = 'fixed'
TIME = 'time'

# parameter names
SIM_COUNT = 'n'
RELIST_COUNT = 'lstg_dur'


# fee constants
ANCHOR_STORE_INSERT = .03

# reject outcomes
AUTO_REJ_OUTCOMES = torch.zeros(len(SLR_OUTCOMES))
AUTO_REJ_OUTCOMES[[7, 8]] = 1  # automatic, rejected
EXP_REJ_OUTCOMES = torch.zeros(len(SLR_OUTCOMES))
EXP_REJ_OUTCOMES[[0, 8, 9]] = 1  # full delay, rejected, expired