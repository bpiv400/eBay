import numpy as np, pandas as pd
from models.datasets.eBayDataset import eBayDataset
from constants import *


class DiscrimDataset(eBayDataset):
    def __init__(self, part, name):
        super().__init__(part, name)


    def __getitem__(self, idx):
        # y is indexed directly
        y = self.d['y'][idx]

        # components of x are indexed using idx_x
        x_idx = self.d['x_idx'][idx]
        x = {k: v[x_idx,:] for k, v in self.d['x'].items()}

        # add directly indexed components to x
        if 'x_arrival' in self.d:
        	x['arrival'] = self.d['x_arrival'][idx,:]

        return y, x