import sys, os, argparse
import torch, torch.optim as optim
import numpy as np, pandas as pd
from datetime import datetime as dt
from compress_pickle import load
from models.simulator.interface import Inputs, run_loop
from models.simulator.model import Simulator
from constants import *


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(description='Training development.')
    parser.add_argument('--model', type=str, 
        help='One of arrival, hist, [delay/con/msg]_[byr/slr].')
    args = parser.parse_args()
    model = args.model

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
    print(sizes)

    # initialize neural net
    simulator = Simulator(model, sizes, dropout=False)
    print(simulator.net)
    print(simulator.loss)

    # initialize optimizer with default learning rate
    optimizer = optim.Adam(simulator.net.parameters())
    print(optimizer)

    # load data
    train = Inputs('small', model)

    # training
    for epoch in range(10):
        print('Epoch %d' % epoch)

        # training
        t0 = dt.now()
        loss = run_loop(simulator, train, optimizer)
        print('\tloss: %d' % loss)

        sec = (dt.now() - t0).total_seconds()
        print('\ttime: %d sec' % sec)

    # # save model
    # torch.save(simulator.net.state_dict(),
    #     MODEL_DIR + 'small/%s.net' % model)