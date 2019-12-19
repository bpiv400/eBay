import torch, torch.nn as nn
from models.nets import FeedForward, LSTM
from models.sample import get_batches
from constants import *


class Model:
    def __init__(self, name, sizes, dropout=False, device='cuda'):
        '''
        Creates a neural net and manages training and validation.
        :param name: string name of model.
        :param sizes: dictionary of data sizes.
        :param dropout: True if using dropout.
        :param device: either 'cuda' or 'cpu'
        '''
        self.name = name
        self.dropout = dropout
        self.device = device

        # recurrent models
        self.isRecurrent = (name == 'arrival') or ('delay' in name)

        # initialize gamma to 0
        self.gamma = 0.0

        # loss function
        if name in ['hist', 'con_slr', 'con_byr']:
            self.loss = nn.CrossEntropyLoss(reduction='sum')
        elif name == 'arrival':
            self.loss = nn.PoissonNLLLoss(log_input=False, reduction='sum')
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction='sum')

        # neural net
        if self.isRecurrent:
            self.net = LSTM(sizes, dropout=dropout).to(device)
        else:
            self.net = FeedForward(sizes, dropout=dropout).to(device)

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


    def get_loss(self, theta, y):
        # binary cross entropy requires float
        if str(self.loss) == "BCEWithLogitsLoss()":
            y = y.float()

        # recurrent models
        if self.isRecurrent:
            mask = y > -1
            theta = theta[mask]
            y = y[mask]

        # zero-inflated poisson for arrival model
        if self.name == 'arrival':
            pi = torch.sigmoid(theta[:,0])
            lamb = torch.exp(theta[:,1])

            pi0 = pi[y == 0]
            loss0 = -torch.sum(torch.log(pi0 + (1-pi0) * torch.exp(-lamb[y == 0])))
            loss1 = -torch.sum(torch.log(1-pi[y > 0]))
            loss2 = self.loss(lamb[y > 0], y[y > 0])

            return loss0 + loss1 + loss2

        # feed-forward models
        return self.loss(theta, y)


    def move_to_device(self, b):
        '''
        :param b: batch as a dictionary.
        :return: batch as a dictionary with components on self.device.
        '''
        if self.device != 'cpu':
            b['x'] = {k: v.to(self.device) for k, v in b['x'].items()}
            b['y'] = b['y'].to(self.device)
            if 'x_time' in b:
                b['x_time'] = b['x_time'].to(self.device)


    def simulate(self, d):
        if self.isRecurrent:
            return self.net(d['x'], d['x_time']).squeeze()
        return self.net(d['x']).squeeze()


    def run_batch(self, b, optimizer):
        '''
        Loops over examples in batch, calculates loss.
        :param b: batch of examples from DataLoader.
        :param optimizer: instance of torch.optim.
        :return: scalar loss.
        '''
        isTraining = optimizer is not None  # train / eval mode

        # call forward on model
        self.net.train(isTraining)
        theta = self.simulate(b)

        # calculate loss
        loss = self.get_loss(theta, b['y'])

        # add in regularization penalty and step down gradients
        if isTraining:
            if self.gamma > 0:
                factor = float(torch.sum(b['y'] > -1)) / data.N_labels
                loss = loss + self.get_penalty(factor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    def run_loop(self, data, optimizer=None):
        '''
        Calculates loss on epoch, steps down gradients if training.
        :param data: Inputs object.
        :param optimizer: instance of torch.optim.
        :return: scalar loss.
        '''
        batches = get_batches(data, optimizer is not None)

        # loop over batches, calculate log-likelihood
        loss = 0.0
        for b in batches:
            self.move_to_device(b)
            loss += self.run_batch(b, optimizer)

        return loss


    def predict_theta(self, data):
        '''
        Collects output of forward call.
        :param data: Inputs object
        :return: tensor of model output.
        '''
        self.net.train(False)
        batches = get_batches(data, False)
        
        # predict theta
        theta = []
        for b in batches:
            self.move_to_device(b)
            theta.append(self.simulate(b))

        return torch.cat(theta)