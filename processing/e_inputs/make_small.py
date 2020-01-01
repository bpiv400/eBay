import sys, os, argparse
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from collections import OrderedDict
from processing.processing_consts import *
from constants import *
from featnames import *


if __name__ == '__main__':
 	# extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='model name')
    name = parser.parse_args().name

    d = load(INPUT_DIR + 'train_models/{}'.format(name))

    # initialize dictionary to save
    small = {}
 
    # randomly select N_SMALL indices
    N = np.shape(d['periods'])[0] if 'periods' in d else np.shape(d['y'])[0]
    v = np.arange(N)
    np.random.shuffle(v)
    idx = v[:N_SMALL]
 
    # recurrent models
    if 'periods' in d:
        small['periods'] = d['periods'][idx]
 
        small['y'], small['tf'] = {}, {}
        for i in idx:
            if i in d['y']:
                small['y'][i] = d['y'][i]
            if i in d['tf']:
                small['tf'][i] = d['tf'][i]
 
    # feed forward
    else:
        y = d['y'][idx]
 
    # directly subsample
    for k in d.keys():
        if not isinstance(d[k], dict):
            small[k] = d[k][idx]
 
    # loop through x
    if 'x' in d:
        small['x'] = {k: v[idx, :] for k, v in d['x'].items()}
    else:
    	x = load(PARTS_DIR + 'train_models/x_lstg.gz')
    	x = {k: v.iloc[idx] for k, v in x.items()}
    	for v in x.values():
			assert np.all(v.index == small['periods'].index)
		dump(x, PARTS_DIR + 'small/x_lstg.gz')
 
    # save dictionary
    dump(small, INPUT_DIR + 'small/{}.gz'.format(name))
 