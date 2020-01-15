import sys, os, argparse
import torch, torch.optim as optim
import numpy as np, pandas as pd
from datetime import datetime as dt
from compress_pickle import load
from model.datasets.eBayDataset import eBayDataset
from model.Model import Model
from model.model_consts import *
from constants import INPUT_DIR, PARAMS_PATH, MODEL_DIR


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='model name')
    name = parser.parse_args().name

    # load model sizes
    print('Loading parameters')
    sizes = load(INPUT_DIR + 'sizes/{}.pkl'.format(name))
    print('Sizes: {}'.format(sizes))

    # load parameters
    params = load(PARAMS_PATH)
    print('Parameters: {}'.format(params))

    # initialize neural net
    model = Model(name, sizes, params)
    print(model.net)
    print(model.loss.__name__)

    # initialize optimizer with default learning rate
    optimizer = optim.Adam(model.net.parameters(), lr=0.001)
    print(optimizer)

    # load data
    data = eBayDataset('small', name, sizes)

    # smoothing parameters
    if (name == 'arrival') or ('delay' in name):
        model.smoothing = 100

    # training
    for epoch in range(10):
        print('Epoch %d:' % epoch)

        # training
        print('\tTraining:')
        t0 = dt.now()
        loss = model.run_loop(data, optimizer)
        print('\t\tTotal time: {} seconds'.format(
            (dt.now() - t0).total_seconds()))
        print('\t\tloss: %d' % loss)
        
        # # testing
        # print('\tTesting:')
        # t0 = dt.now()
        # with torch.no_grad():
        #     loss_test = model.run_loop(data)
        # print('\t\tTotal time: {} seconds'.format(
        #     (dt.now() - t0).total_seconds()))
        # print('\t\tloss: %d' % loss_test)

    # save model
    torch.save(model.net.state_dict(), 
        MODEL_DIR + 'small/{}.net'.format(name))