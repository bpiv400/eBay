import numpy as np
import torch
import torch.nn as nn
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from train.eBayDataset import eBayDataset
from train.train_consts import FTOL, LOG_DIR, LNLR0, LNLR1, LNLR_FACTOR
from train.Model import Model
from train.Sample import get_batches
from train.loss import time_loss, taylor_softmax_loss
from constants import MODEL_DIR


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
        """
        # save parameters to self
        self.name = name
        self.dev = dev
        self.device = device

        # loss function
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        print(self.loss)

        # load datasets
        self.train = eBayDataset(train_part, name)
        self.test = eBayDataset(test_part, name)

        # turn-specific baserates
        assert name in ['first_con', 'con_slr']     # add con_byr later
        if name == 'first_con':


    def train_model(self, gamma=0):
        """
        Public method to train model.
        :param gamma: scalar regularization parameter for variational dropout.
        """
        # experiment id
        expid = dt.now().strftime('%y%m%d-%H%M')

        # initialize writer
        if not self.dev:
            writer = SummaryWriter(
                LOG_DIR + '{}/{}'.format(self.name, expid))
        else:
            writer = None

        # tune initial learning rate
        lnlr, model = self._tune_lr(writer=writer, gamma=gamma)

        # path to save model
        model_path = MODEL_DIR + '{}/{}.net'.format(self.name, expid)

        # save model
        if not self.dev:
            torch.save(model.net.state_dict(), model_path)

        # initialize optimizer and scheduler
        optimizer = Adam(model.net.parameters(), lr=np.exp(lnlr))
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
            output = self._run_epoch(model, optimizer,
                                     writer=writer, epoch=epoch)

            # save model
            if not self.dev:
                torch.save(model.net.state_dict(), model_path)

            # reduce learning rate if loss has not meaningfully improved
            scheduler.step(output['loss'])

            # stop training if learning rate is sufficiently small
            if self._get_lnlr(optimizer) < LNLR1:
                break

            # update last, increment epoch
            last = output['loss']
            epoch += 1

        return -output['lnL_test']

    @staticmethod
    def _get_lnlr(optimizer):
        for param_group in optimizer.param_groups:
            return np.log(param_group['lr'])

    def _run_epoch(self, model, optimizer, writer=None, epoch=None):
        # initialize output with log10 learning rate
        output = dict()
        output['lnlr'] = self._get_lnlr(optimizer)

        # train model
        output['loss'] = self._run_loop(self.train, model, optimizer)

        # collect remaining output and print
        output = self._collect_output(model, writer, output, epoch=epoch)

        return output

    def _run_loop(self, data, model, optimizer=None):
        """
        Calculates loss on epoch, steps down gradients if training.
        :param data: Inputs object.
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
            b['x'] = {k: v.to(self.device) for k, v in b['x'].items()}
            b['y'] = b['y'].to(self.device)

            # increment loss
            loss += self._run_batch(b, model, optimizer)

            # increment gpu time
            gpu_time += (dt.now() - t1).total_seconds()

        # print timers
        prefix = 'training' if is_training else 'validation'
        print('\t{0:s} GPU time: {1:.1f} seconds'.format(prefix, gpu_time))
        print('\ttotal {0:s} time: {1:.1f} seconds'.format(prefix,
                                                           (dt.now() - t0).total_seconds()))

        return loss

    def _run_batch(self, b, model, optimizer):
        """
        Loops over examples in batch, calculates loss.
        :param b: batch of examples from DataLoader.
        :param optimizer: instance of torch.optim.
        :return: scalar loss.
        """
        is_training = optimizer is not None  # train / eval mode

        # call forward on model
        model.net.train(is_training)
        theta = model.net(b['x']).squeeze()

        # calculate loss
        loss = self.loss(theta, b['y'].squeeze())

        # add in regularization penalty and step down gradients
        if is_training:
            if model.penalized:
                penalty = model.get_penalty()
                factor = len(b['y']) / len(self.train)
                # print(loss.item(), penalty * factor)
                loss = loss + penalty * factor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    def _tune_lr(self, writer=None, gamma=0):
        models, loss = [], []
        for lnlr in LNLR0:
            # initialize model and optimizer
            models.append(self._initialize_model(gamma))
            optimizer = Adam(models[-1].net.parameters(), lr=np.exp(lnlr))

            # print to console
            if len(models) == 1:
                print(models[-1].net)
            print('Tuning with lnlr of {}'.format(lnlr))

            # run model for one epoch
            loss.append(self._run_loop(self.train, models[-1], optimizer))
            print('\tloss: {}'.format(loss[-1]))

        # best learning rate and model
        idx = int(np.argmin(loss))
        lnlr = LNLR0[idx]
        model = models[idx]

        # initialize output with log10 learning rate
        output = {'lnlr': lnlr, 'loss': loss[idx]}

        # collect remaining output and print
        print('Epoch 0')
        self._collect_output(model, writer, output)

        # return lnlr of smallest loss and corresponding model
        return lnlr, model

    def _initialize_model(self, gamma=0):
        # uninitialized weights
        model = Model(self.name, gamma=gamma, device=self.device)

        # load model weights from pretrained model
        if model.dropout:
            model.net.load_state_dict(
                torch.load(self.pretrained_path), strict=False)

        return model

    def _collect_output(self, model, writer, output, epoch=0):
        # calculate log-likelihood on validation set
        with torch.no_grad():
            loss_train = self._run_loop(self.train, model)
            output['lnL_train'] = -loss_train / self.train.N
            loss_test = self._run_loop(self.test, model)
            output['lnL_test'] = -loss_test / self.test.N

        # save output to tensorboard writer
        for k, v in output.items():
            print('\t{}: {}'.format(k, v))
            if writer is not None:
                writer.add_scalar(k, v, epoch)

        return output
