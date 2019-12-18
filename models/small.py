import sys, os, argparse
import torch, torch.optim as optim
import numpy as np, pandas as pd
from datetime import datetime as dt
from compress_pickle import load
from models.inputs import eBayDataset
from models.model import Model
from constants import *


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, 
        help='One of arrival, hist, [delay/con/msg]_[byr/slr].')
    args = parser.parse_args()
    name = args.name

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, name))
    print(sizes)

    # initialize neural net
    model = Model(name, sizes)
    print(model.net)
    print(model.loss)

    # initialize optimizer with default learning rate
    optimizer = optim.Adam(model.net.parameters())
    print(optimizer)

    # load data
    train = eBayDataset('small', name)

    # training
    for epoch in range(10):
        print('Epoch %d' % epoch)

        # training
        t0 = dt.now()
        loss = model.run_loop(train, optimizer)
        print('\tloss: %d' % loss)

        sec = (dt.now() - t0).total_seconds()
        print('\ttime: %d sec' % sec)

    # # save model
    # torch.save(model.net.state_dict(),
    #     MODEL_DIR + 'small/%s.net' % name)