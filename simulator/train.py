import sys, os, pickle, argparse
import torch
import numpy as np, pandas as pd
from compress_pickle import load
from datetime import datetime as dt
from torch.utils.data import DataLoader
from simulator.interface import *
from simulator.model import Simulator
from constants import *


def run_loop(simulator, data, isTraining=False):
    # collate function
    f = collateRNN if data.isRecurrent else collateFF

    # minibatch size
    mbsize = simulator.mbsize if isTraining else MBSIZE_VALIDATION

    # sampler
    sampler = Sample(data, mbsize, isTraining)

    # load batches
    batches = DataLoader(data, batch_sampler=sampler,
        collate_fn=f, num_workers=NUM_WORKERS, pin_memory=True)

    # loop over batches, calculate log-likelihood
    lnL = 0
    for batch in batches:
        lnL += simulator.run_batch(batch, isTraining)

    return lnL / data.N_labels


def train_model(simulator, train, test, filename):
    # loop over epochs, record log-likelihood
    for i in range(EPOCHS):
        t0 = dt.now()

        # training loop
        print('Training epoch %d:' % (i+1))
        lnL_train = run_loop(simulator, train, isTraining=True)

        # calculate log-likelihood on validation set
        print('\tValidating on holdout.')
        with torch.no_grad():
            lnL_test = run_loop(simulator, test)

        # epoch duration
        dur = np.round((dt.now() - t0).seconds)

        # write statistics to file
        f = open(filename, 'a')
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
    print('%s: %d' % (model, paramsid))

    # load model sizes
    print('Loading parameters')
    sizes = pickle.load(
        open('%s/inputs/sizes/%s.pkl' % (PREFIX, model), 'rb'))
    print(sizes)

    # load experiment parameters
    params = pd.read_csv('%s/inputs/params.csv' % PREFIX, 
        index_col=0).loc[paramsid].to_dict()
    print(params)

    # initialize neural net
    simulator = Simulator(model, params, sizes, 
        device='cuda' if torch.cuda.is_available() else 'cpu')
    print(simulator.net)

    # load datasets
    train = load('%s/inputs/train_models/%s.gz' % (PREFIX, model))
    test = load('%s/inputs/train_rl/%s.gz' % (PREFIX, model))

    # create outfile
    filename = SUMMARY_DIR + '%s/%d.csv' % (model, paramsid)
    f = open(filename, 'w')
    f.write('epoch,seconds,lnL_train,lnL_holdout\n')
    f.close()

    # train model
    train_model(simulator, train, test, filename)
