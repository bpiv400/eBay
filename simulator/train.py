import sys, os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from torch.utils.data import DataLoader
from simulator.interface import *
from simulator.simulator import Simulator
from constants import *

EPOCHS = 1000


def run_loop(model, simulator, data, isTraining=False):
    # collate function
    f = collateRNN if data.isRecurrent else collateFF

    # load batches
    batches = DataLoader(data, batch_sampler=Sample(data, isTraining),
        collate_fn=f, num_workers=NUM_WORKERS, pin_memory=True)

    # loop over batches, calculate log-likelihood
    lnL = 0
    for batch in batches:
        lnL += simulator.run_batch(batch, isTraining)
        print(lnL)

    return lnL


def train_model(simulator, train, test, outfile):
    # loop over epochs, record log-likelihood
    for i in range(EPOCHS):
        t0 = dt.now()

        # training loop
        run_loop(model, simulator, train, isTraining=True)

        # calculate log-likelihood on training and test
        lnL_train = run_loop(model, simulator, train)
        lnL_test = run_loop(model, simulator, test)

        # epoch duration
        dur = np.round((dt.now() - t0).seconds)

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

    # load model sizes
    print('Loading parameters')
    sizes = pickle.load(
        open('%s/inputs/sizes/%s.pkl' % (PREFIX, model), 'rb'))
    print(sizes)

    # load neural net parameters
    params = pd.read_csv('%s/inputs/params.csv' % PREFIX, 
        index_col=0).loc[paramsid].to_dict()
    print(params)
    
    # initialize neural net
    simulator = Simulator(model, params, sizes, 
        device='cuda' if torch.cuda.is_available() else 'cpu')
    print(simulator.net)

    # create datasets
    train = Inputs('train_models', model)
    test = Inputs('train_rl', model)

    # before training, calculate log-likelihood on training and test
    t0 = dt.now()
    lnL0_train = run_loop(model, simulator, train)
    lnL0_test = run_loop(model, simulator, test)
    dur = np.round((dt.now() - t0).seconds)

    # create outfile
    outfile = 'outputs/cluster/%s_%d.csv' % (model, paramsid)
    f = open(outfile, 'w')
    f.write('epoch,seconds,lnL_train,lnL_holdout\n')
    f.write('%d,%d,%.4f,%.4f\n' % (0, dur, lnL0_train, lnL0_test))
    f.close()

    # train model
    train_model(simulator, train, test, outfile)
    