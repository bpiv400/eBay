import sys, os
import torch
import numpy as np, pandas as pd
from compress_pickle import load
from nets.FeedForward import FeedForward
from constants import *
from utils import load_model

NAME = 'arrival'

# load 
d = load(INPUT_DIR + '{}/{}.gz'.format(VALIDATION, NAME))
idx = load(INDEX_DIR + '{}/{}.gz'.format(VALIDATION, NAME))

# reconstruct
y = pd.Series(d['y'], index=idx, name='interval')
x = {k: pd.DataFrame(v, index=idx) for k, v in d['x'].items()}

# restrict to first thread
y = y.xs(1, level='thread')
x = {k: v.xs(1, level='thread') for k, v in x.items()}

# create model
net = load_model(NAME)

# x to torch
x = {k: torch.tensor(v.values).float() for k, v in x.items()}

# predictions
theta = net(x)

