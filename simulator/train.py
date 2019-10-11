import sys, os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from torch.utils.data import DataLoader
from interface import *
from simulator import Simulator

sys.path.append('repo/')
from constants import *


def get_dataloader(model, outcome):
    data = Inputs('train_models', model, outcome)
    if model == 'arrival':
        f = collateFF
    else:
        f = collateRNN

    loader = DataLoader(data, batch_sampler=Sample(data),
        num_workers=0, collate_fn=f)

    return loader


def train_model(simulator, epochs):
    # initialize array for log-likelihoods by epoch
    lnL = []

    # loop over epochs, record log-likelihood
    for i in range(epochs):
        time0 = dt.now()

        # get data and data loader
        loader = get_dataloader(simulator.model, simulator.outcome)

        # loop over batches
        lnL_i = 0
        for j, batch in enumerate(loader):
            # batch is [data, idx]
            lnL_i += simulator.run_batch(*batch)
            print(lnL_i)

        # append log-likelihood to list
        lnL.append(lnL_i)

        # print log-likelihood and duration
        print('Epoch %d: lnL: %1.4f. (%dsec)' %
            (i+1, lnL[-1], (dt.now() - time0).seconds))

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
    paramsid = args.id

    # model folder
    file = lambda x: 'data/inputs/%s/%s_%s.pkl' % (x, model, outcome)

    # load inputs to model
    print('Loading parameters')
    sizes = pickle.load(open(file('sizes'), 'rb'))
    params = pickle.load(open(file('params'), 'rb')).loc[paramsid]
    print(params)

    # initialize neural net
    simulator = Simulator(model, outcome, params, sizes)
    print(simulator.net)

    # number of epochs
    epochs = int(np.ceil(UPDATES * MBSIZE / sizes['N']))

    # train model
    print('Training: %d epochs.' % epochs)
    lnL_train, duration = train_model(simulator, epochs)

    # save model
    torch.save(simulator.net.state_dict(), folder + str(args.id) + '.pt')


    #### NEED TO UPDATE WITH DATALOADER

    # holdout
    holdout = pickle.load(open(folder + 'train_rl.pkl', 'rb'))
    loss_holdout, _ = simulator.evaluate_loss(holdout, train=False)
    print('Holdout lnL: %.4f.' % -loss_holdout.item())

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
    