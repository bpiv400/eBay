from constants import BYR_DROP
from featnames import X_LSTG, LSTG, STORE
from utils import load_featnames

# thread features appended to x_lstg
DAYS_SINCE_START_IND = -5

# fixed features
featnames = load_featnames(X_LSTG)[LSTG]
featnames = [c for c in featnames if c not in BYR_DROP]
STORE_IND = featnames.index(STORE)
START_PRICE_PCTILE_IND = featnames.index('start_price_pctile')
