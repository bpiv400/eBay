from constants import META_PATH
import pandas as pd
from featnames import META, START_PRICE


META_LABELS = pd.read_csv(META_PATH).set_index(META).squeeze()

price = ['(0-{})'.format(SPLIT_VALS[START_PRICE][1])]
for i in range(1, len(SPLIT_VALS[START_PRICE]) - 2):
    low = SPLIT_VALS[START_PRICE][i]
    high = SPLIT_VALS[START_PRICE][i + 1]
    price.append('[{}-{})'.format(low, high))
price.append('[{}-1000]'.format(SPLIT_VALS[START_PRICE][-2]))

SPLIT_LABELS = {META: [meta[i] for i in SPLIT_VALS[META]],
                START_PRICE: price_labels}

SPLIT_YLABELS = {META: None, START_PRICE: 'Buy-it-now price ($)'}
