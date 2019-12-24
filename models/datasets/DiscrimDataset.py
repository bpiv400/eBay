import numpy as np, pandas as pd
from compress_pickle import load
from models.datasets.eBayDataset import eBayDataset
from constants import *


class DiscrimDataset(eBayDataset):
    def __init__(self, part, name):
        super().__init__(part, name)


    def __getitem__(self, idx):
        # y and x are indexed directly
        y = self.d['y'][idx]
        x = {k: v[idx,:] for k, v in self.d['x'].items()}

        return y, x