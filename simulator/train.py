import sys, os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from torch.utils.data import DataLoader
from interface import *
from simulator import Simulator

sys.path.append('repo/')
from constants import *


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
    model = args.model
    outcome = args.outcome

    # model folder
    folder = '%s/%s/%s/' % (MODEL_DIR, model, outcome)

    # load inputs to model
    print('Loading parameters')
    sizes = pickle.load(open(folder + 'sizes.pkl', 'rb'))
    params = pd.read_csv(folder + 'params.csv', index_col=0).loc[args.id]

    # data loader
    dataset = Inputs('train_models', model, outcome)
    sampler = Sample(dataset)
    if model == 'arrival':
        f = collateFF

    loader = DataLoader(dataset, batch_sampler=sampler,
        num_workers=1, collate_fn=f)

    # initialize neural net
    simulator = Simulator(args.model, args.outcome, train, params, sizes)
    print(simulator.net)

    # train model
    print('Training: %d epochs.' % simulator.epochs)
    lnL_train, duration = train_model(simulator)

    # holdout
    holdout = pickle.load(open(folder + 'train_rl.pkl', 'rb'))
    loss_holdout, _ = simulator.evaluate_loss(holdout, train=False)
    print('Holdout lnL: %.4f.' % -loss_holdout.item())

    # save model
    torch.save(simulator.net.state_dict(), folder + str(args.id) + '.pt')

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
    