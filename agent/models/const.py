from constants import BYR_DROP, PCTILE_DIR
from featnames import X_LSTG, LSTG, STORE, START_PRICE
from utils import load_featnames, unpickle

# thread features appended to x_lstg
DAYS_SINCE_START_IND = -5

# fixed features
featnames = load_featnames(X_LSTG)[LSTG]
featnames = [c for c in featnames if c not in BYR_DROP]
STORE_IND = featnames.index(STORE)
START_PRICE_PCTILE_IND = featnames.index('start_price_pctile')

# percentiles of start price
START_PRICE_PCTILES = unpickle(PCTILE_DIR + '{}.pkl'.format(START_PRICE))
