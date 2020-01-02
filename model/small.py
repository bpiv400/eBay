import sys, os, argparse
import torch, torch.optim as optim
import numpy as np, pandas as pd
from datetime import datetime as dt
from compress_pickle import load
from model.FeedForwardDataset import FeedForwardDataset
from model.RecurrentDataset import RecurrentDataset
from model.Model import Model
from model.model_consts import *
from constants import *


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='model name')
    name = parser.parse_args().name

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, name))
    print(sizes)

    # initialize neural net
    model = Model(name, sizes, batchnorm=False, dropout=False)
    print(model.net)
    print(model.loss)

    # initialize optimizer with default learning rate
    optimizer = optim.Adam(model.net.parameters())
    print(optimizer)

    # load data
    dataset = RecurrentDataset if 'x_time' in sizes else FeedForwardDataset
    train = dataset('small', name)

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