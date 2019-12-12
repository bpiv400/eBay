import sys, os, argparse
import torch, torch.optim as optim
import numpy as np, pandas as pd
from compress_pickle import load, dump
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from simulator.interface import Inputs, run_loop
from simulator.model import Simulator
from constants import *


def train_model(simulator, optimizer, writer, train, test, paramsid):
    kl = paramsid * KL_INT  # KL regularization coefficient
    print('KL coefficient: %1.2f' % kl)

    for epoch in range(30):
        print('Epoch %d' % epoch)
        output = {}

        # training loop
        t0 = dt.now()
        output['loss'] = run_loop(simulator, train, kl, optimizer)
        output['penalty'] = simulator.get_penalty().item()
        output['sec_train'] = (dt.now() - t0).total_seconds()

        # calculate log-likelihood on training and validation sets
        for name in ['train', 'test']:
            data = globals()[name]
            with torch.no_grad():
                t0 = dt.now()
                loss = run_loop(simulator, data)
                output['lnL_' + name] = -loss / data.N_labels

        # save output to tensorboard writer and print to console
        for k, v in output.items():
            writer.add_scalar(k, v, epoch)
            if np.abs(v) > 1:
                print('\t%s: %d' % (k, v))
            else:
                print('\t%s: %9.4f' % (k, v))

        # save model
        path = MODEL_DIR + '%s/%d_post_%d.net' \
            % (simulator.model, paramsid, epoch)
        torch.save(simulator.net.state_dict(), path)


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str)
    parser.add_argument('--id', type=int)
    args = parser.parse_args()
    model = args.model
    paramsid = args.id

    # load model sizes
    print('Loading parameters')
    sizes = load('%s/inputs/sizes/%s.pkl' % (PREFIX, model))
    print(sizes)

    # initialize neural net and loss function
    simulator = Simulator(model, sizes)

    # initialize optimizer
    optimizer = optim.Adam(simulator.net.parameters())
    
    # print modules
    print(simulator.net)
    print(simulator.loss)
    print(optimizer)

    # load pretrained network
    simulator.net.load_state_dict(torch.load(
        MODEL_DIR + '%s/%d_pre.net' % (model, paramsid)))

    # print penalty for pretrained network
    print('Penalty for pretrained network: %9.0f' \
        % simulator.get_penalty())

    # load datasets
    train = Inputs('train_models', model)
    test = Inputs('train_rl', model)

    # initialize tensorboard writer
    writer = SummaryWriter(LOG_DIR + '%s/%d' % (model, paramsid))

    # train model
    model_path = MODEL_DIR + '%s/%d_post.net' % (model, paramsid)
    epochs, output = train_model(simulator, optimizer, 
        writer, train, test, paramsid)
    writer.close()

    # save result
    output['epochs'] = epochs
    dump(output, EXPS_DIR + '%s.pkl' % model)
    