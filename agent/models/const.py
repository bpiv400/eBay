from featnames import X_LSTG, LSTG, STORE, START_PRICE
from utils import load_featnames

# thread features appended to x_lstg
DAYS_SINCE_START_IND = -5

# fixed features
featnames = load_featnames(X_LSTG)[LSTG]
STORE_IND = featnames.index(STORE)
START_PRICE_IND = featnames.index(START_PRICE)
