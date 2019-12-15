import sys, os, argparse
import torch, torch.optim as optim
import numpy as np, pandas as pd
from compress_pickle import load, dump
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from models.simulator.interface import Inputs, run_loop
from models.simulator.model import Simulator
from constants import *


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str, 
        help='One of arrival, delay_byr, delay_slr, con_byr, con_slr.')
    args = parser.parse_args()
    model = args.model

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
    print(sizes)

    # initialize neural net without dropout
    simulator = Simulator(model, sizes, dropout=False)

    # initialize learning rate and optimizer
    lr = LR0
    optimizer = optim.Adam(simulator.net.parameters(), lr=lr)

    # print modules
    print(simulator.net)
    print(simulator.loss)
    print(optimizer)

    # load datasets
    train = Inputs('train_models', model)
    test = Inputs('train_rl', model)

    # initialize tensorboard writer
    writer = SummaryWriter(LOG_DIR + '%s/pretrain' % model)

    # training without variational dropout
    epoch, last = 0, np.inf
    while True:
        print('Epoch %d' % epoch)
        output = {'lr': lr}

        # training
        t0 = dt.now()
        output['loss'] = run_loop(simulator, train, optimizer)

        # calculate log-likelihood on training and validation sets
        with torch.no_grad():
            loss_train = run_loop(simulator, train)
            output['lnL_train'] = -loss_train / train.N_labels

            loss_test = run_loop(simulator, test)
            output['lnL_test'] = -loss_test / test.N_labels

        # save output to tensorboard writer and print to console
        for k, v in output.items():
            writer.add_scalar(k, v, epoch)
            if k in ['loss', 'penalty', 'largest']:
                print('\t\t%s: %d' % (k, v))
            elif 'lnL' in k:
                print('\t\t%s: %1.4f' % (k, v))
            else:
                print('\t\t%s: %2.2f' % (k, v))
        print('\t\ttime: %d sec' % (dt.now() - t0).total_seconds())

        # stopping condition
        if output['loss'] > last * FTOL:
            if lr == 1e-5:
                break
            else:
                lr /= 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        # increment epochs and update last
        epoch += 1
        last = output['loss']

    # save model
    torch.save(simulator.net.state_dict(),
        MODEL_DIR + 'pretrain/%s.net' % model)