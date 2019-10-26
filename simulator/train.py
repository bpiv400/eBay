import sys, os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from torch.utils.data import DataLoader
from interface import *
from simulator import Simulator
from constants import *

EPOCHS = 1000

def run_loop(model, simulator, data, isTraining):
    # collate function
    f = collateFF if model == 'hist' else collateRNN

    # data loader
    loader = DataLoader(data, batch_sampler=Sample(data),
        collate_fn=f, num_workers=0, pin_memory=True)

    # loop over batches, calculate loss
    lnL = 0
    for batch in loader:
        lnL += simulator.run_batch(*batch, isTraining)

    return lnL


def train_model(simulator, train, test, outfile):
    # loop over epochs, record loss
    for i in range(EPOCHS):
        t0 = dt.now()

        # training loop
        lnL_train = run_loop(model, simulator, train, True)

        # training duration
        dur = np.round((dt.now() - t0).seconds)

        # test loop
        lnL_test = run_loop(model, simulator, test, False)

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

    # create outfile
    outfile = 'outputs/cluster/%s_%d.csv' % (model, paramsid)
    f = open(outfile, 'w')
    f.write('epoch,seconds,lnL_train,lnL_holdout\n')
    f.close()

    # train model
    train_model(simulator, train, test, outfile)
    