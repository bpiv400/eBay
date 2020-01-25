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
net = load_model(NAME).to('cuda')

# x to torch
x = {k: torch.tensor(v.values).float() for k, v in x.items()}

# batches
v = np.array([i for i in range(len(y))])
batches = np.array_split(v, 1 + len(v) // 2048)

# probability of no arrival
p0 = np.array([], dtype='float32')
for b in batches:
	x_b = {k: v[b, :].to('cuda') for k, v in x.items()}
	p = torch.exp(torch.nn.functional.log_softmax(net(x_b), dim=1))
	p0 = np.append(p0, p[:, -1].cpu().numpy())
