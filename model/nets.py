import torch, torch.nn as nn
from collections import OrderedDict
from model.LSTM import LSTM
from constants import EMBEDDING_GROUPS


class VariationalDropout(nn.Module):
    def __init__(self, N):
        super(VariationalDropout, self).__init__()
        
        self.log_alpha = nn.Parameter(torch.Tensor(N))
        self.log_alpha.data.fill_(-3)
        
    def kl_reg(self):
        # calculate KL divergence
        kl = 0.63576 * (torch.sigmoid(1.8732 + 1.48695 * self.log_alpha) - 1) \
            - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        return -torch.sum(kl)
    
    def forward(self, x):
        # additive N(0, alpha * x^2) noise
        if self.training:
            sigma = torch.sqrt(torch.exp(self.log_alpha)) * torch.abs(x)
            eps = torch.normal(torch.zeros_like(x), torch.ones_like(x))
            return x + sigma * eps
        else:
            return x


class Layer(nn.Module):
    def __init__(self, N_in, N_out, params):
        '''
        :param N_in: scalar number of input weights.
        :param N_out: scalar number of output weights.
        :param params: dictionary of neural net parameters.

        '''
        super(Layer, self).__init__()

        # initialize layer
        self.layer = nn.ModuleList([nn.Linear(N_in, N_out)])

        # batch normalization
        if params['batchnorm']:
            self.layer.append(
                nn.BatchNorm1d(N_out, affine=params['affine']))

        # activation function
        self.layer.append(nn.ReLU(inplace=True))

        # dropout
        if params['dropout']:
            self.layer.append(VariationalDropout(N_out))


    def forward(self, x):
        '''
        x: tensor of shape [mbsize, N_in]
        '''
        for m in self.layer:
            x = m(x)
        return x


class Stack(nn.Module):
    def __init__(self, N, params, layers=1):
        '''
        :param N: scalar number of input and output weights.
        :param layers: scalar number of layers to stack.
        :param params: dictionary of neural net parameters.
        '''
        super(Stack, self).__init__()

        # sequence of modules
        self.stack = nn.ModuleList([])
        for _ in range(layers):
            self.stack.append(Layer(N, N, params))

        
    def forward(self, x):
        '''
        x: tensor of shape [mbsize, N_in]
        '''
        for m in self.stack:
            x = m(x)
        return x


class Embedding(nn.Module):
    def __init__(self, counts, params):
        '''
        :param counts: dictionary of scalar input sizes.
        :param params: dictionary of neural net parameters.
        '''
        super(Embedding, self).__init__()

        # first stack of layers: N to N
        self.layer1 = nn.ModuleDict()
        for k, v in counts.items():
            self.layer1[k] = Stack(v, params,
                layers=params['layers_embedding'])

        # second layer: concatenation
        N = sum(counts.values())
        self.layer2 = Stack(N, params,
            layers=params['layers_embedding'])


    def forward(self, x):
        '''
        x: OrderedDict() with a superset of the keys in self.k
        '''
        # first layer
        l = []
        for k, m in self.layer1.items():
            l.append(m(x[k]))

        # concatenate
        x = torch.cat(l, dim=1)

        # pass through second embedding
        return self.layer2(x)


class FullyConnected(nn.Module):
    def __init__(self, N_in, N_out, params):
        '''
        N_in: scalar number of input weights.
        N_out: scalar number of output parameters.
        :param params: dictionary of neural net parameters.
        '''
        super(FullyConnected, self).__init__()
        # intermediate layer
        self.seq = nn.ModuleList([Layer(N_in, params['hidden'], params)])

        # fully connected network
        self.seq.append(Stack(params['hidden'], params,
            layers=params['layers_full']-2))

        # last layer has no dropout
        self.seq.append(Stack(params['hidden'], params))

        # output layer
        self.seq.append(nn.Linear(params['hidden'], N_out))


    def forward(self, x):
        '''
        x: tensor of shape [mbsize, N_in]
        '''
        for m in self.seq:
            x = m(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, sizes, params, toRNN=False):
        '''
        :param sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        :param params: dictionary of neural net parameters.
        :param toRNN: True if FeedForward initialized hidden state of recurrent network
        '''
        super(FeedForward, self).__init__()

        # expand embeddings
        groups = EMBEDDING_GROUPS.copy()
        if 'offer1' in sizes['x']:
            groups['offer'] = ['lstg'] \
                + [k for k in sizes['x'].keys() if 'offer' in k]
        elif 'arrival' in sizes['x']:
            groups['arrival'] = ['lstg', 'arrival']

        # embeddings
        d, total = OrderedDict(), 0
        for name, group in groups.items():
            counts = {k: v for k, v in sizes['x'].items() if k in group}
            d[name] = Embedding(counts, params)
            total += sum(counts.values())
        self.nn0 = nn.ModuleDict(d)

        # fully connected
        self.nn1 = FullyConnected(total, 
            params['hidden'] if toRNN else sizes['out'], params)


    def forward(self, x):
        '''
        x: OrderedDict()
        '''
        l = []
        for k in self.nn0.keys():
            l.append(self.nn0[k](x))
        return self.nn1(torch.cat(l, dim=1))


class Recurrent(nn.Module):
    def __init__(self, sizes, params):
        '''
        :param sizes: dictionary of scalar input sizes; sizes['x'] is an OrderedDict
        :param params: dictionary of neural net parameters.
        '''
        super(Recurrent, self).__init__()

        # initial hidden nodes
        self.h0 = FeedForward(sizes, params, toRNN=True)
        self.c0 = FeedForward(sizes, params, toRNN=True)

        # rnn layer
        self.rnn = LSTM(input_size=sizes['x_time'],
                        hidden_size=params['hidden'],
                        batch_first=True,
                        affine=params['affine'])

        # output layer
        self.output = nn.Linear(params['hidden'], sizes['out'])


    def forward(self, x, x_time):
        '''
        :param x: OrderedDict()
        :param x_time: tensor of shape [mbsize, INTERVAL_COUNTS[outcome], sizes['x_time']]
        '''
        # initialize hidden state
        hidden = (self.h0(x).unsqueeze(dim=0), 
            self.c0(x).unsqueeze(dim=0))

        # update hidden state recurrently
        theta, _ = self.rnn(x_time, hidden)

        # # pad
        # theta, _ = nn.utils.rnn.pad_packed_sequence(
        #     theta, total_length=len(x_time.batch_sizes), 
        #     batch_first=True)

        # output layer: (batch_size, seq_len, N_output)
        return self.output(theta).squeeze()


    def init(self, x=None):
        return (self.h0(x).unsqueeze(dim=0), 
            self.c0(x).unsqueeze(dim=0))


    def step(self, x_time=None, hidden=None, x=None):
        """
        Executes 1 step of the recurrent network

        :param x_time:
        :param x:
        :param hidden:
        :return:
        """
        if hidden is None:
            hidden = self.init(x=x)
        theta, hidden = self.rnn(x_time.unsqueeze(0), hidden)
        # output layer: (seq_len, batch_size, N_out)
        return self.output(theta), hidden
