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

    # randomly select indices
    v = np.arange(np.shape(d['y'])[0])
    np.random.shuffle(v)
    idx = v[:N_SMALL]

    # create dictionary
    small = {k: v[idx] for k, v in d.items()}

    # save dictionary
    dump(small, INPUT_DIR + 'small/{}.gz'.format(name))
 