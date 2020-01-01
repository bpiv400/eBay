import sys, os, argparse, math
import numpy as np
import torch, torch.optim as optim
from compress_pickle import load, dump
from model.Model import Model
from model.model_utils import get_dataset
from model.model_consts import *
from constants import *


def training_loop(model, train, test, expid):
    # initialize optimizer
    loglr = LOGLR0
    optimizer = optim.Adam(model.net.parameters(), lr=math.pow(10, loglr))
    print(optimizer)

    # initialize tensorboard writer
    writer = SummaryWriter('{}{}/{}'.format(LOG_DIR, expid))

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
        torch.save(model.net.state_dict(), 
            '{}/{}.net'.format(model_dir, expid))

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
    print('Parameters: {}'.format(sizes))

    # initialize model
    model = Model(name, sizes, dropout=False)
    print(model.net)

    # load datasets
    dataset = get_dataset(name)
    print('Loading training data')
    train = dataset('train_models', name)
    print('Loading validation data')
    test = dataset('train_rl', name)

    # experiment number
    expid = 0
    while True:
        if os.path.isdir(LOG_DIR + '{}/{}'.format(name, expid)):
            expid += 1
        else:
            break

    # training loop
    training_loop(model, train, test, expid)