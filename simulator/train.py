import sys
sys.path.append('repo/')
sys.path.append('repo/simulator/')
import os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from simulator import Simulator
from constants import *
from parsing_funcs import *
#from utils import *


def train_model(simulator):
    time0 = dt.now()

    # initialize array for log-likelihoods by epoch
    lnL = []

    # loop over epochs, record log-likelihood
    for i in range(simulator.epochs):
        start = dt.now()

        # iterate over minibatches
        lnL.append(simulator.run_epoch())

        # print log-likelihood and duration
        print('Epoch %d: lnL: %1.4f. (%dsec)' %
            (i+1, lnL[-1], (dt.now() - start).seconds))

    # return loss history and total duration
    return lnL, (dt.now() - time0).seconds


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str,
        help='One of: arrival, byr, slr.')
    parser.add_argument('--outcome', type=str, help='Outcome to predict.')
    parser.add_argument('--id', type=int, help='Experiment ID.')
    args = parser.parse_args()

    # extract parameters from CSV, training data
    params = parse_params(args.model, args.outcome, args.id)
    print(params)

    # create inputs to model
    print('Creating model inputs')
    train, featnames = process_inputs(
        'train_models', args.model, args.outcome)

    # get data size parameters
    sizes = get_sizes(args.model, args.outcome, params, train)

    # initialize neural net
    simulator = Simulator(args.model, args.outcome, train, params, sizes)
    print(simulator.net)

    # train model
    print('Training: %d epochs.' % simulator.epochs)
    lnL_train, duration = train_model(simulator)

    # holdout
    print('Evaluating on holdout set')
    holdout, _ = process_inputs('train_rl', args.model, args.outcome)
    loss_holdout, _ = simulator.evaluate_loss(holdout, train=False)

    # save model
    folder = MODEL_DIR + args.model + '/' + args.outcome + '/' 
    torch.save(simulator.net.state_dict(), folder + str(args.id) + '.pt')

    # save featnames and sizes once per model
    if args.id == 1:
        pickle.dump([featnames, sizes], open(folder + 'info.pkl', 'wb'))

    # save log-likelihood and duration to results CSV
    path = folder + 'results.csv'
    if os.path.exists(path):
        T = pd.read_csv(path, index_col=0)
    else:
        T = pd.DataFrame(index=pd.Index([], name='expid'))
    T.loc[args.id, 'lnL_holdout'] = -loss_holdout.item()
    T.loc[args.id, 'sec_per_epoch'] = duration / simulator.epochs
    for i in range(simulator.epochs):
        T.loc[args.id, 'lnL_train_' + str(i+1)] = lnL_train[i]

    T.to_csv(path)
    