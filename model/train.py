import sys, os, argparse, math
import numpy as np
import torch, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from compress_pickle import load, dump
from model.Model import Model
from model.datasets.FeedForwardDataset import FeedForwardDataset
from model.datasets.ArrivalDataset import ArrivalDataset
from model.datasets.DelayDataset import DelayDataset
from model.model_consts import *
from constants import INPUT_DIR, MODEL_DIR, PARAMS_PATH


def training_loop(model, train, test, writer_path, model_path):
    # initialize optimizer
    loglr = LOGLR0
    optimizer = optim.Adam(model.net.parameters(), lr=math.pow(10, loglr))
    print(optimizer)

    epoch, last = 0, np.inf
    while True:
        print('\tEpoch %d' % epoch)
        output = {'loglr': loglr}

        # train model
        output['loss'] = model.run_loop(train, optimizer)
        output['lnL_train'] = -output['loss'] / train.N

        # calculate log-likelihood on validation set
        with torch.no_grad():
            loss_test = model.run_loop(test)
            output['lnL_test'] = -loss_test / test.N

        # initialize tensorboard writer in first epoch
        if epoch == 0:
            writer = SummaryWriter(writer_path)

        # save output to tensorboard writer
        for k, v in output.items():
            writer.add_scalar(k, v, epoch)

        # save model
        torch.save(model.net.state_dict(), model_path)

        # reduce learning rate until convergence
        if output['loss'] > FTOL * last:
            if loglr == LOGLR1:
                break
            else:
                loglr -= LOGLR_INC
                for param_group in optimizer.param_groups:
                    param_group['lr'] = math.pow(10, loglr)

        # increment epochs and update last
        epoch += 1
        last = output['loss']

    writer.close()


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    name = parser.parse_args().name

    # load model sizes
    sizes = load(INPUT_DIR + 'sizes/{}.pkl'.format(name))
    print('Sizes: {}'.format(sizes))

    # load parameters
    params = load(PARAMS_PATH)
    print('Parameters: {}'.format(params))

    # initialize model
    model = Model(name, sizes, params)
    print(model.net)
    print(model.loss.__name__)

    # load datasets
    print('Loading data')
    if name == 'arrival':
        train = ArrivalDataset('train_models', name, sizes)
        test = ArrivalDataset('train_rl', name, sizes)
    elif 'delay' in name:
        train = DelayDataset('train_models', name, sizes)
        test = DelayDataset('train_rl', name, sizes)
    else:
        train = FeedForwardDataset('train_models', name, sizes)
        test = FeedForwardDataset('train_rl', name, sizes)

    # experiment number
    expid = 0
    while True:
        if os.path.isdir(LOG_DIR + '{}/{}'.format(name, expid)):
            expid += 1
        else:
            break

    # paths
    writer_path = LOG_DIR + '{}/{}'.format(name, expid)
    model_path = MODEL_DIR + '{}/{}.net'.format(name, expid)

    # training loop
    training_loop(model, train, test, writer_path, model_path)