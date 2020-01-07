import sys, os, argparse, math
import numpy as np
import torch, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from compress_pickle import load, dump
from model.Model import Model
from model.FeedForwardDataset import FeedForwardDataset
from model.RecurrentDataset import RecurrentDataset
from model.model_consts import *
from constants import INPUT_DIR, MODEL_DIR, PARAMS_PATH


def training_loop(model, train, test, writer, model_path):
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

        # calculate log-likelihood on training and validation sets
        with torch.no_grad():
            loss_train = model.run_loop(train)
            output['lnL_train'] = -loss_train / train.N_labels

            loss_test = model.run_loop(test)
            output['lnL_test'] = -loss_test / test.N_labels

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

    # load datasets
    print('Loading data')
    if 'x_time' in sizes:
        train = RecurrentDataset('train_models', name, sizes)
        test = RecurrentDataset('train_rl', name, sizes)
    else:
        train = FeedForwardDataset('train_models', name)
        test = FeedForwardDataset('train_rl', name)

    # experiment number
    expid = 0
    while True:
        if os.path.isdir(LOG_DIR + '{}/{}'.format(name, expid)):
            expid += 1
        else:
            break

    # initialize tensorboard writer
    writer = SummaryWriter(LOG_DIR + '{}/{}'.format(name, expid))

    # model path
    model_path = MODEL_DIR + '{}/{}.net'.format(name, expid)

    # training loop
    training_loop(model, train, test, writer, model_path)