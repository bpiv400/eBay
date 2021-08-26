import os
import torch
from torch.nn.functional import log_softmax, nll_loss
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from sim.EBayDataset import EBayDataset
from nets.FeedForward import FeedForward
from sim.Sample import get_batches
from paths import LOG_DIR, MODEL_DIR
from featnames import CENSORED_MODELS, BYR_HIST_MODEL
from utils import load_sizes

LR_FACTOR = 0.1  # multiply learning rate by this factor when training slows
LR0 = 1e-3  # initial learning rate
LR1 = 1e-7  # stop training when learning rate is lower than this
FTOL = 1e-2  # decrease learning rate when relative improvement in loss is less than this
AMSGRAD = True  # use AMSgrad version of ADAM if True


class SimTrainer:
    """
    Trains a model until convergence.

    Public methods:
        * train: trains the initialized model under given parameters.
    """

    def __init__(self, name=None):
        """
        :param name: string model name.
        """
        # save parameters to self
        self.name = name

        # boolean for different loss functions
        self.use_time_loss = name in CENSORED_MODELS
        self.use_count_loss = name == BYR_HIST_MODEL

        # load model size parameters
        self.sizes = load_sizes(name)
        print(self.sizes)

        # load datasets
        self.train = EBayDataset(name=name, train=True)
        self.valid = EBayDataset(name=name, train=False)

    def train_model(self, dropout=(0.0, 0.0), log=True):
        """
        Public method to train model.
        :param dropout: pair of dropout rates, one for last embedding, one for fully connected.
        :param log: True for Tensorboard logging.
        """
        # experiment id
        dtstr = dt.now().strftime('%y%m%d-%H%M')
        levels = [int(d * 10) for d in dropout]
        expid = '{}_{}_{}'.format(dtstr, levels[0], levels[1])

        # initialize writer
        if log:
            writer_path = LOG_DIR + '{}/{}'.format(self.name, expid)
            writer = SummaryWriter(writer_path)
        else:
            writer = None

        # path to save model
        model_dir = MODEL_DIR + '{}/'.format(self.name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + '{}.net'.format(expid)

        # initialize neural net, optimizer and scheduler
        net = FeedForward(self.sizes, dropout=dropout).to('cuda')
        optimizer = Adam(net.parameters(), lr=LR0, amsgrad=AMSGRAD)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=LR_FACTOR,
                                                   patience=0,
                                                   threshold=FTOL)
        print(net)
        print(optimizer)

        # training loop
        epoch, lnl_test0 = 0, None
        while True:
            # run one epoch
            print('Epoch {}'.format(epoch))
            output = self._run_epoch(net, 
                                     optimizer=optimizer,
                                     writer=writer,
                                     epoch=epoch)

            # set 0th-epoch log-likelihood
            if epoch == 0:
                lnl_test0 = output['lnL_test']

            # save model
            if log:
                torch.save(net.state_dict(), model_path)

            # reduce learning rate if loss has not meaningfully improved
            scheduler.step(output['loss'])

            # stop training if learning rate is sufficiently small
            if self._get_lr(optimizer) < LR1:
                break

            # stop training if holdout objective hasn't improved in 9 epochs
            if epoch >= 8 and output['lnL_test'] < lnl_test0:
                break

            # increment epoch
            epoch += 1

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
        loss, gpu_time = 0., 0.
        t0 = dt.now()
        for b in batches:
            t1 = dt.now()

            # move to device
            for key, value in b.items():
                if type(value) is dict:
                    b[key] = {k: v.to('cuda') for k, v in value.items()}
                else:
                    b[key] = value.to('cuda')

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
        lnl = torch.sum(lnq[arrival, y[arrival]])

        # non-arrivals
        cens = y < 0
        y_cens = y[cens]
        q_cens = torch.exp(lnq[cens, :])
        for i in range(q_cens.size()[0]):
            lnl += torch.log(torch.sum(q_cens[i, y_cens[i]:]))

        return -lnl

    @staticmethod
    def _count_loss(theta, y):
        """
        Implements beta negative binomial distribution with r=1
        :param float tensor theta: parameters from model
        :param int tensor y: count of previous best-offer listings
        :return: negative log-likelihood
        """
        # transformations
        pi = torch.sigmoid(theta[:, 0])
        params = torch.exp(theta[:, 1:])

        # split by y
        idx0, idx1 = y == 0, y > 0
        pi0, pi1 = pi[idx0], pi[idx1]
        a0, a1 = params[idx0, 0], params[idx1, 0]
        b0, b1 = params[idx0, 1], params[idx1, 1]
        r0, r1 = params[idx0, 2] + 1, params[idx1, 2] + 1
        y1 = y[idx1].float()

        # zeros
        if len(pi0) > 0:
            # lnl = torch.sum(torch.log(pi0 + (1-pi0) * a0 / (a0 + b0)))
            f0 = torch.exp(torch.lgamma(r0 + a0)
                           + torch.lgamma(a0 + b0)
                           - torch.lgamma(a0)
                           - torch.lgamma(a0 + b0 + r0))
            lnl = torch.sum(torch.log(pi0 + (1-pi0) * f0))
        else:
            lnl = 0.

        # non-zeros
        if len(y1) > 0:
            lnl += torch.sum(torch.log(1-pi1)
                             + torch.lgamma(r1 + y1)
                             + torch.lgamma(r1 + a1)
                             + torch.lgamma(b1 + y1)
                             + torch.lgamma(a1 + b1)
                             - torch.lgamma(r1)
                             - torch.lgamma(a1 + b1 + r1 + y1)
                             - torch.lgamma(a1)
                             - torch.lgamma(b1)
                             - torch.lgamma(y1 + 1))

        return -lnl

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

        if self.use_count_loss:
            loss = self._count_loss(theta, b['y'])
        else:
            # softmax
            if theta.size()[1] == 1:
                theta = torch.cat((torch.zeros_like(theta), theta), dim=1)
            lnq = log_softmax(theta, dim=-1)

            # calculate loss
            if self.use_time_loss:
                loss = self._time_loss(lnq, b['y'])
            else:
                loss = nll_loss(lnq, b['y'], reduction='sum')

        # step down gradients
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    def _collect_output(self, net, writer, output, epoch=0):
        # calculate log-likelihood on validation set
        with torch.no_grad():
            loss_train = self._run_loop(self.train, net)
            loss_test = self._run_loop(self.valid, net)
            output['lnL_train'] = -loss_train / self.train.N
            output['lnL_test'] = -loss_test / self.valid.N

        # save output to tensorboard writer
        for k, v in output.items():
            print('\t{}: {}'.format(k, v))
            if writer is not None:
                writer.add_scalar(k, v, epoch)

        return output
