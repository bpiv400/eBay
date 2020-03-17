import numpy as np
import torch
from torch.nn.functional import log_softmax, nll_loss
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from train.EBayDataset import EBayDataset
from train.train_consts import FTOL, LOG_DIR, LNLR0, LNLR1, LNLR_FACTOR, INT_DROPOUT
from nets.FeedForward import FeedForward
from train.Sample import get_batches
from constants import MODEL_DIR
from utils import load_sizes


class Trainer:
    """
    Trains a model until convergence.

    Public methods:
        * train: trains the initialized model under given parameters.
    """

    def __init__(self, name, train_part, test_part, dev=False, device='cuda'):
        """
        :param name: string model name.
        :param train_part: string partition name for training data.
        :param test_part: string partition name for holdout data.
        :param dev: True for development.
        :param device: 'cuda' or 'cpu'.
        """
        # save parameters to self
        self.name = name
        self.dev = dev
        self.device = device

        # boolean for time loss
        self.is_delay = 'delay' in name or name == 'next_arrival'

        # dropout rate to be set later
        self.dropout = None

        # load model size parameters
        self.sizes = load_sizes(name)
        print(self.sizes)

        # load datasets
        self.train = EBayDataset(train_part, name)
        self.test = EBayDataset(test_part, name)

    def train_model(self, dropout=(0.0, 0.0)):
        """
        Public method to train model.
        :param dropout: pair of dropout rates, one for last embedding, one for fully connected.
        """
        # save dropout to self
        self.dropout = dropout

        # experiment id
        dtstr = dt.now().strftime('%y%m%d-%H%M')
        levels = [int(d * INT_DROPOUT) for d in dropout]
        expid = '{}_{}_{}'.format(dtstr, levels[0], levels[1])

        # initialize writer
        if not self.dev:
            writer_path = LOG_DIR + '{}/{}'.format(self.name, expid)
            writer = SummaryWriter(writer_path)
        else:
            writer = None

        # tune initial learning rate
        lnlr, net, lnL_test0 = self._tune_lr(writer)

        # path to save model
        model_path = MODEL_DIR + '{}/{}.net'.format(self.name, expid)

        # save model
        if not self.dev:
            torch.save(net.state_dict(), model_path)

        # initialize optimizer and scheduler
        optimizer = Adam(net.parameters(), lr=np.exp(lnlr))
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=np.exp(LNLR_FACTOR),
                                                   patience=0,
                                                   threshold=FTOL)
        print(optimizer)

        # training loop
        epoch, last = 1, np.inf
        while True:
            # run one epoch
            print('Epoch {}'.format(epoch))
            output = self._run_epoch(net, 
                                     optimizer=optimizer,
                                     writer=writer,
                                     epoch=epoch)

            # save model
            if not self.dev:
                torch.save(net.state_dict(), model_path)

            # reduce learning rate if loss has not meaningfully improved
            scheduler.step(output['loss'])

            # stop training if learning rate is sufficiently small
            if self._get_lnlr(optimizer) < LNLR1:
                break

            # stop training if holdout objective hasn't improved in 12 epochs
            if epoch >= 11 and output['lnL_test'] < lnL_test0:
                break

            # update last, increment epoch
            last = output['loss']
            epoch += 1

        return -output['lnL_test']

    @staticmethod
    def _get_lnlr(optimizer):
        for param_group in optimizer.param_groups:
            return np.log(param_group['lr'])

    def _run_epoch(self, net, optimizer=None, writer=None, epoch=None):
        # initialize output with log10 learning rate
        output = dict()
        output['lnlr'] = self._get_lnlr(optimizer)

        # train model
        output['loss'] = self._run_loop(self.train, net, optimizer)

        # collect remaining output and print
        output = self._collect_output(net, writer, output, epoch=epoch)

        return output

    def _run_loop(self, data, net, optimizer=None):
        """
        Calculates loss on epoch, steps down gradients if training.
        :param data: Inputs object.
        :param net: FeedForward instance.
        :param optimizer: instance of torch.optim.
        :return: scalar loss.
        """
        is_training = optimizer is not None
        batches = get_batches(data, is_training=is_training)

        # loop over batches, calculate log-likelihood
        loss = 0.0
        gpu_time = 0.0
        t0 = dt.now()
        for b in batches:
            t1 = dt.now()

            # move to device
            for key, value in b.items():
                if type(value) is dict:
                    b[key] = {k: v.to(self.device) for k, v in value.items()}
                else:
                    b[key] = value.to(self.device)

            # increment loss
            loss += self._run_batch(b, net, optimizer)

            # increment gpu time
            gpu_time += (dt.now() - t1).total_seconds()

        # print timers
        prefix = 'training' if is_training else 'validation'
        print('\t{0:s} GPU time: {1:.1f} seconds'.format(prefix, gpu_time))
        print('\ttotal {0:s} time: {1:.1f} seconds'.format(prefix,
                                                           (dt.now() - t0).total_seconds()))

        return loss

    @staticmethod
    def _time_loss(lnq, y):
        # arrivals have positive y
        arrival = y >= 0
        lnL = torch.sum(lnq[arrival, y[arrival]])

        # non-arrivals
        cens = y < 0
        y_cens = y[cens]
        q_cens = torch.exp(lnq[cens, :])
        for i in range(q_cens.size()[0]):
            lnL += torch.log(torch.sum(q_cens[i, y_cens[i]:]))

        return -lnL

    def _run_batch(self, b, net, optimizer):
        """
        Loops over examples in batch, calculates loss.
        :param b: batch of examples from DataLoader.
        :param net: FeedForward instance.
        :param optimizer: instance of torch.optim.
        :return: scalar loss.
        """
        is_training = optimizer is not None  # train / eval mode

        # call forward on model
        net.train(is_training)
        theta = net(b['x'])
        if theta.size()[1] == 1:
            theta = torch.cat((torch.zeros_like(theta), theta), dim=1)

        # softmax
        lnq = log_softmax(theta, dim=-1)

        # calculate loss
        if self.is_delay:
            loss = self._time_loss(lnq, b['y'])
        else:
            loss = nll_loss(lnq, b['y'], reduction='sum')

        # add in regularization penalty and step down gradients
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    def _tune_lr(self, writer=None):
        nets, loss = [], []
        for lnlr in LNLR0:
            # initialize model and optimizer
            nets.append(FeedForward(self.sizes, dropout=self.dropout).to(self.device))
            optimizer = Adam(nets[-1].parameters(), lr=np.exp(lnlr))
 
            # print to console
            if len(nets) == 1:
                print(nets[-1])
            print('Tuning with lnlr of {}'.format(lnlr))
 
            # run model for one epoch
            loss.append(self._run_loop(self.train, nets[-1], optimizer))
            print('\tloss: {}'.format(loss[-1]))
 
        # best learning rate and model
        idx = int(np.argmin(loss))
        lnlr = LNLR0[idx]
        net = nets[idx]
 
        # initialize output with log10 learning rate
        output = {'lnlr': lnlr, 'loss': loss[idx]}
 
        # collect remaining output and print
        print('Epoch 0')
        output = self._collect_output(net, writer, output)
 
        # return lnlr of smallest loss and corresponding model
        return lnlr, net, output['lnL_test']

    def _collect_output(self, net, writer, output, epoch=0):
        # calculate log-likelihood on validation set
        with torch.no_grad():
            loss_train = self._run_loop(self.train, net)
            output['lnL_train'] = -loss_train / self.train.N
            loss_test = self._run_loop(self.test, net)
            output['lnL_test'] = -loss_test / self.test.N

        # save output to tensorboard writer
        for k, v in output.items():
            print('\t{}: {}'.format(k, v))
            if writer is not None:
                writer.add_scalar(k, v, epoch)

        return output
