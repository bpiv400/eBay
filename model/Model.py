import torch, torch.nn as nn
from datetime import datetime as dt
from model.nets import FeedForward
from model.Sample import get_batches
from model.model_consts import *
from constants import *


class Model:
    def __init__(self, name, sizes, params, device='cuda'):
        '''
        Creates a neural net and manages training and validation.
        :param name: string name of model.
        :param sizes: dictionary of data sizes.
        :param params: dictionary of neural net parameters.
        :param device: either 'cuda' or 'cpu'
        '''
        self.dropout = params['dropout']
        self.device = device

        # initialize gamma to 0
        self.gamma = 0.0

        # loss function
        if name in ['hist', 'con_slr', 'con_byr']:
            self.loss = nn.CrossEntropyLoss(reduction='sum')
        elif 'msg' in name:
            self.loss = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            self.loss = self._ArrivalTimeLoss

        # neural net
        self.net = FeedForward(sizes, params).to(device)


    def set_gamma(self, gamma):
        if gamma > 0:
            if not self.dropout:
                error('Gamma cannot be positive without dropout layers.')
            self.gamma = gamma


    def get_penalty(self, factor=1):
        penalty = torch.tensor(0.0, device=self.device)
        for m in self.net.modules():
            if hasattr(m, 'kl_reg'):
                penalty += m.kl_reg()
        return self.gamma * factor * penalty


    def get_alpha_stats(self, threshold=9):
        above, total, largest = 0.0, 0.0, 0.0
        for m in self.net.modules():
            if hasattr(m, 'log_alpha'):
                alpha = torch.exp(m.log_alpha)
                largest = max(largest, torch.max(alpha).item())
                total += alpha.size()[0]
                above += torch.sum(alpha > threshold).item()
        return above / total, largest


    def run_loop(self, data, optimizer=None):
        '''
        Calculates loss on epoch, steps down gradients if training.
        :param data: Inputs object.
        :param optimizer: instance of torch.optim.
        :return: scalar loss.
        '''
        batches = get_batches(data, 
            isTraining=optimizer is not None)

        # loop over batches, calculate log-likelihood
        loss = 0.0
        gpu_time = 0.0
        for b in batches:
            t0 = dt.now()
            self._move_to_device(b)
            loss += self._run_batch(b, optimizer)
            gpu_time += (dt.now() - t0).total_seconds()

        print('\t\tGPU time: {} seconds'.format(gpu_time))

        return loss


    def predict_theta(self, data):
        '''
        Collects output of forward call.
        :param data: Inputs object
        :return: tensor of model output.
        '''
        self.net.train(False)
        batches = get_batches(data, isTraining=False)
        
        # predict theta
        theta = []
        for b in batches:
            self._move_to_device(b)
            theta.append(self.net(b['x']))

        return torch.cat(theta)


    def _get_loss(self, theta, y):
        # binary cross entropy requires float
        if str(self.loss) == "BCEWithLogitsLoss()":
            y = y.float()

        return self.loss(theta.squeeze(), y.squeeze())


    def _move_to_device(self, b):
        '''
        :param b: batch as a dictionary.
        :return: batch as a dictionary with components on self.device.
        '''
        if self.device != 'cpu':
            b['x'] = {k: v.to(self.device) for k, v in b['x'].items()}
            b['y'] = b['y'].to(self.device)


    def _run_batch(self, b, optimizer):
        '''
        Loops over examples in batch, calculates loss.
        :param b: batch of examples from DataLoader.
        :param optimizer: instance of torch.optim.
        :return: scalar loss.
        '''
        isTraining = optimizer is not None  # train / eval mode

        # call forward on model
        self.net.train(isTraining)
        theta = self.net(b['x'])

        # calculate loss
        loss = self._get_loss(theta, b['y'])

        # add in regularization penalty and step down gradients
        if isTraining:
            if self.gamma > 0:
                factor = float(torch.sum(b['y'] > -1)) / data.N_labels
                loss = loss + self.get_penalty(factor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    @staticmethod
    def _ArrivalTimeLoss(theta, y):
        # class probabilities
        lnp = nn.functional.log_softmax(theta, dim=-1)

        # arrivals have positive y
        arrival = y >= 0
        lnL = torch.sum(lnp[arrival, y[arrival]])

        # non-arrivals
        cens = y < 0
        y_cens = y[cens]
        p_cens = torch.exp(lnp[cens, :])
        for i in range(p_cens.size()[0]):
            lnL += torch.log(torch.sum(p_cens[i, y_cens[i]:]))

        return -lnL
