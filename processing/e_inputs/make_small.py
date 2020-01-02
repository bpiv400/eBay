import argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from collections import OrderedDict
from processing.processing_consts import N_SMALL
from constants import INPUT_DIR, PARTS_DIR


if __name__ == '__main__':
 	# extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='model name')
    name = parser.parse_args().name

    # load full dictionary
    print('Loading data')
    d = load(INPUT_DIR + 'train_models/{}.pkl'.format(name))

    # recurrent model
    isRecurrent = 'periods' in d

    # initialize dictionary to save
    small = {}

    # recurrent models
    if isRecurrent:
        # randomly sample
        done = False
        while not done:
            small['periods'] = d['periods'].sample(N_SMALL)
            N = small['periods'].unique()
            done = np.all([(small['periods'] == n).count() > 1 for n in N])
        idx = small['periods'].index

        # other elements of d
        for k, v in d.items():
            if k != 'periods':
                if len(d[k].index.names) == 1:
                    small[k] = d[k].reindex(index=idx)
                else:
                    small[k] = d[k].reindex(index=idx, level='lstg')

        # lstg features
        x = load(PARTS_DIR + 'train_models/x_lstg.gz')
        x = {k: v.reindex(index=idx) for k, v in x.items()}
        dump(x, PARTS_DIR + 'small/x_lstg.gz')
 
    # feed forward
    else:
        N = np.shape(d['periods'])[0] if isRecurrent else np.shape(d['y'])[0]
        v = np.arange(N)
        np.random.shuffle(v)
        idx = v[:N_SMALL]

        # listing features
        small['x'] = {k: v[idx, :] for k, v in d['x'].items()}
 
        # directly subsample
        for k in d.keys():
            if not isinstance(d[k], dict):
                small[k] = d[k][idx]
 
    # save dictionary
    dump(small, INPUT_DIR + 'small/{}.pkl'.format(name))
 