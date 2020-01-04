import argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_consts import N_SMALL
from constants import INPUT_DIR


if __name__ == '__main__':
 	# extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='model name')
    name = parser.parse_args().name

    # load full dictionary
    d = load(INPUT_DIR + 'train_models/{}.gz'.format(name))

    # initialize dictionary to save
    small = {}

    # recurrent models
    if 'periods' in d:
        # randomly sample
        small['periods'] = d['periods'].sample(N_SMALL)
        idx = small['periods'].index

        # other elements of d
        for k, v in d.items():
            if k != 'periods':
                if len(d[k].index.names) == 1:
                    small[k] = d[k].reindex(index=idx)
                else:
                    small[k] = d[k].reindex(index=idx, level='lstg')

        # lstg features
        small['x'] = {k: v.reindex(index=idx) for k, v in d['x'].items()}
 
    # feed forward
    else:
        v = np.arange(np.shape(d['y'])[0])
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
 