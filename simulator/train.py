import sys, os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from torch.utils.data import DataLoader
from interface import *
from simulator import Simulator
from constants import *

EPOCHS = 1000
TRAIN_PART = 'train_models'
TEST_PART = 'train_rl'


def get_dataloader(model, part):
    data = Inputs(part, model)
    if model == 'hist':
        f = collateFF
    else:
        f = collateRNN

    loader = DataLoader(data, batch_sampler=Sample(data),
        num_workers=0, collate_fn=f, pin_memory=True)

    return loader


def train_model(simulator, outfile):
    # initialize array for log-likelihoods by epoch
    lnL = []

    # loop over epochs, record log-likelihood
    for i in range(EPOCHS):
        time0 = dt.now()

        # training loop
        train = get_dataloader(simulator.model, TRAIN_PART)
        lnL_train = 0
        for batch in train:
            lnL_train += simulator.run_batch(*batch, train=True)

        # training duration
        dur = np.round((dt.now() - t0).seconds)

        # test loop
        test = get_dataloader(simulator.model, TEST_PART)
        lnL_test = 0
        for batch in test:
            lnL_test += simulator.run_batch(*batch, train=False)

        # print log-likelihood and duration
        print('Epoch %d: lnL: %1.4f. (%dsec)' %
            (i+1, lnL[-1], (dt.now() - time0).seconds))

        # write to file
        f = open(outfile, 'a')
        f.write('%d,%d,%.4f,%.4f\n' % (i+1, dur, lnL_train, lnL_test))
        f.close()


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str, 
        help='One of arrival, delay_byr, delay_slr, con_byr, con_slr.')
    parser.add_argument('--id', type=int, help='Experiment ID.')
    args = parser.parse_args()
    model = args.model
    paramsid = args.id

    # model folder
    file = lambda x: '%s/inputs/%s/%s.pkl' % (PREFIX, x, model)

    # load model sizes and parameters
    print('Loading parameters')
    sizes = pickle.load(open(file('sizes'), 'rb'))
    print(sizes)
    params = pd.read_csv('%s/inputs/params.csv' % PREFIX, 
        index_col=0).loc[paramsid].to_dict()
    print(params)
    
    # initialize neural net
    simulator = Simulator(model, params, sizes, 
        device='cuda' if torch.cuda.is_available() else 'cpu')
    print(simulator.net)

    # create outfile
    outfile = 'outputs/cluster/%s_%d.csv' % (model, paramsid)
    f = open(outfile, 'w')
    f.write('epoch,seconds,lnL_train,lnL_holdout\n')
    f.close()

    # train model
    train_model(simulator, outfile)
    