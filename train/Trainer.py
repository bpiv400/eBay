import numpy as np
import torch
from torch.nn.functional import log_softmax, nll_loss
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from train.EBayDataset import EBayDataset
from nets.FeedForward import FeedForward
from train.Sample import get_batches
from nets.const import LAYERS_EMBEDDING
from train.const import FTOL, LR0, LR1, LR_FACTOR, INT_DROPOUT
from constants import MODEL_DIR, LOG_DIR, DELAY_MODELS, \
    INTERARRIVAL_MODEL, INIT_VALUE_MODELS, MODEL_NORM
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

        # boolean for different loss functions
        self.is_delay = name in DELAY_MODELS or name == INTERARRIVAL_MODEL
        self.is_init_value = name in INIT_VALUE_MODELS

        # load model size parameters
        self.sizes = load_sizes(name)
        print(self.sizes)

        # load datasets
        self.train = EBayDataset(train_part, name)
        self.test = EBayDataset(test_part, name)

    def train_model(self, dropout=(0.0, 0.0), norm=MODEL_NORM):
        """
        Public method to train model.
        :param dropout: pair of dropout rates, one for last embedding, one for fully connected.
        :param layers0: number of layers in first embedding
        :param norm: one of ['batch', 'layer', 'weight']
        """
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
        lr, net, lnl_test0 = self._tune_lr(writer=writer,
                                           dropout=dropout,
                                           norm=norm)

        # path to save model
        model_path = MODEL_DIR + '{}/{}.net'.format(self.name, expid)

        # save model
        if not self.dev:
            torch.save(net.state_dict(), model_path)

        # initialize optimizer and scheduler
        optimizer = Adam(net.parameters(), lr=lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=LR_FACTOR,
                                                   patience=0,
                                                   threshold=FTOL)
        print(optimizer)

        # training loop
        epoch = 1
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
            if self._get_lr(optimizer) < LR1:
                break

            # stop training if holdout objective hasn't improved in 12 epochs
            if epoch >= 11 and output['lnL_test'] < lnl_test0:
                break

            # increment epoch
            epoch += 1

        return -output['lnL_test']

    @staticmethod
    def _get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _run_epoch(self, net, optimizer=None, writer=None, epoch=None):
        # initialize output with log10 learning rate
        output = dict()
        output['lr'] = self._get_lr(optimizer)

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

        # softmax
        if theta.size()[1] == 1:
            theta = torch.cat((torch.zeros_like(theta), theta), dim=1)
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

    def _tune_lr(self, writer=None, dropout=None, norm=None):
        nets, loss = [], []
        for lr in LR0:
            # initialize model and optimizer
            nets.append(FeedForward(self.sizes,
                                    dropout=dropout,
                                    norm=norm).to(self.device))
            optimizer = Adam(nets[-1].parameters(), lr=lr)
 
            # print to console
            if len(nets) == 1:
                print(nets[-1])
            print('Tuning with lr of {}'.format(lr))
 
            # run model for one epoch
            loss.append(self._run_loop(self.train, nets[-1], optimizer))
            print('\tloss: {}'.format(loss[-1]))
 
        # best learning rate and model
        idx = int(np.argmin(loss))
        lr = LR0[idx]
        net = nets[idx]
 
        # initialize output with log10 learning rate
        output = {'lr': lr, 'loss': loss[idx]}
 
        # collect remaining output and print
        print('Epoch 0')
        output = self._collect_output(net, writer, output)
 
        # return lr of smallest loss and corresponding model
        return lr, net, output['lnL_test']

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
