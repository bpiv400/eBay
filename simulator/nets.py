import torch, torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, params, sizes, toRNN=False):
        # super constructor
        super(FeedForward, self).__init__()

        # save dictionary of input feat sizes
        self.k = sizes['x']

        # activation function
        self.f = nn.ReLU()

        # embeddings layer(s)
        d = {}
        for k, v in self.k.items():
            l = []
            for i in range(2):
                l += [nn.Linear(v, v), self.f]
            d[k] = nn.ModuleList(l)
        self.embedding = nn.ModuleDict(d)

        # intermediate layers
        self.seq = nn.ModuleList(
            [nn.Linear(sum(self.k.values()), params['hidden']), self.f])
        for i in range(params['layers']-1):
            self.seq.append(nn.Dropout(p=params['dropout'] / 10))
            self.seq.append(
                nn.Linear(params['hidden'], params['hidden']))
            self.seq.append(self.f)

        # output layer
        if toRNN:
            self.seq.append(nn.Linear(params['hidden'], params['hidden']))
        else:
            self.seq.append(nn.Linear(params['hidden'], sizes['out']))

    def forward(self, x):
        '''
        x: OrderedDict() with same keys as self.k
        '''
        # separate embeddings for input feature components
        l = []
        for key in self.k.keys():
            x_k = x[key]
            for m in self.embedding[key]:
                x_k = m(x_k)
            l.append(x_k)

        # concatenate
        x = torch.cat(l, dim=1)

        # fully connected layers with dropout
        for m in self.seq:
            x = m(x)
        return x

    def simulate(self, x):
        return self.forward(x)


class LSTM(nn.Module):
    def __init__(self, params, sizes):

        # super constructor
        super(LSTM, self).__init__()

        # save parameters to self
        self.steps = sizes['steps']

        # initial hidden nodes
        self.h0 = FeedForward(params, sizes, toRNN=True)
        self.c0 = FeedForward(params, sizes, toRNN=True)

        # rnn layer
        self.rnn = nn.LSTM(input_size=sizes['time'],
                           hidden_size=int(params['hidden']),
                           batch_first=True)

        # output layer
        self.output = nn.Linear(params['hidden'], sizes['out'])

    # output discrete weights and parameters of continuous components
    def forward(self, x_fixed, x_time):
        x_fixed = x_fixed.unsqueeze(dim=0)
        theta, _ = self.rnn(x_time, (self.h0(x_fixed), self.c0(x_fixed)))

        # pad
        theta, _ = nn.utils.rnn.pad_packed_sequence(
            theta, total_length=self.steps, batch_first=True)

        # output layer: (batch_size, seq_len, N_output)
        return self.output(theta).squeeze()

    def init(self, x_fixed=None):
        x_fixed.unsqueeze(dim=0)
        hidden = (self.h0(x_fixed), self.c0(x_fixed))
        return hidden

    def simulate(self, x_time, x_fixed=None, hidden=None):
        """

        :param x_time:
        :param x_fixed:
        :param hidden:
        :return:
        """
        if hidden is None:
            x_fixed.unsqueeze(dim=0).repeat(self.layers, 1, 1)
            theta, hidden = self.rnn(x_time, (self.h0(x_fixed), self.c0(x_fixed)))
        else:
            theta, hidden = self.rnn(x_time, hidden)

        # output layer: (seq_len, batch_size, N_out)
        return self.output(theta), hidden
