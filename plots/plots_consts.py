import pandas as pd
from processing.processing_consts import CLEAN_DIR
from assess.assess_consts import SPLIT_VALS
from featnames import META, START_PRICE

# fontsize by plot type
FONTSIZE = {'roc': 16,
            'training': 24,
            'p': 16}

# labels
meta_labels = pd.read_csv(CLEAN_DIR + 'meta.csv',
                          names=[META],
                          squeeze=True)
meta_labels.index += 1
meta_labels = meta_labels.to_dict()

price_labels = ['(0-{})'.format(SPLIT_VALS[START_PRICE][1])]
for i in range(1, len(SPLIT_VALS[START_PRICE]) - 2):
    low = SPLIT_VALS[START_PRICE][i]
    high = SPLIT_VALS[START_PRICE][i + 1]
    price_labels.append('[{}-{})'.format(low, high))
price_labels.append('[{}-1000]'.format(SPLIT_VALS[START_PRICE][-2]))

SPLIT_LABELS = {META: [meta_labels[i] for i in SPLIT_VALS[META]],
                START_PRICE: price_labels}

SPLIT_YLABELS = {META: None, START_PRICE: 'Buy-it-now price ($)'}
