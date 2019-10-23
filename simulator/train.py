import sys, os, pickle, argparse
import torch
import numpy as np, pandas as pd
from datetime import datetime as dt
from torch.utils.data import DataLoader
from interface import *
from simulator import Simulator

sys.path.append('repo/')
from constants import *

LAYERS = 2
HIDDEN = 256


def get_dataloader(model):
    data = Inputs('train_models', model)
    if model == 'hist':
        f = collateFF
    else:
        f = collateRNN

    loader = DataLoader(data, batch_sampler=Sample(data),
        num_workers=0, collate_fn=f, pin_memory=True)

    return loader


def train_model(simulator, epochs):
    # initialize array for log-likelihoods by epoch
    lnL = []

    # loop over epochs, record log-likelihood
    for i in range(epochs):
        time0 = dt.now()

        # get data and data loader
        loader = get_dataloader(simulator.model)

        # loop over batches
        lnL_i = 0
        for j, batch in enumerate(loader):
            lnL_i += simulator.run_batch(*batch)

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
    params = {'layers': LAYERS, 'hidden': HIDDEN}
    print(params)
    
    # initialize neural net
    simulator = Simulator(model, params, sizes, device='cuda')
    print(simulator.net)

    # number of epochs
    epochs = int(np.ceil(UPDATES * MBSIZE / sizes['N']))

    # train model
    print('Training: %d epochs.' % epochs)
    lnL_train, duration = train_model(simulator, epochs)


    #### NEED TO UPDATE

    # save model
    torch.save(simulator.net.state_dict(), folder + str(args.id) + '.pt')

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
    