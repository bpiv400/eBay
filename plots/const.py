import os
import pandas as pd
from constants import CLEAN_DIR
from assess.const import META_LABELS
from featnames import META, START_PRICE

# where to save figures
FIG_DIR = '{}/{}/'.format(os.path.expanduser('~/eBay'), 'figures')

# fontsize by plot type
FONTSIZE = {'roc': 16,
            'training': 24,
            'p': 16}

# # labels
# META_LABELS = pd.read_csv(CLEAN_DIR + 'meta.csv').set_index(META).squeeze()
#
# price = ['(0-{})'.format(SPLIT_VALS[START_PRICE][1])]
# for i in range(1, len(SPLIT_VALS[START_PRICE]) - 2):
#     low = SPLIT_VALS[START_PRICE][i]
#     high = SPLIT_VALS[START_PRICE][i + 1]
#     price.append('[{}-{})'.format(low, high))
# price.append('[{}-1000]'.format(SPLIT_VALS[START_PRICE][-2]))

# SPLIT_LABELS = {META: [meta[i] for i in SPLIT_VALS[META]],
#                 START_PRICE: price_labels}
#
# SPLIT_YLABELS = {META: None, START_PRICE: 'Buy-it-now price ($)'}
