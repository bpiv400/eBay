import torch
import torch.nn as nn
from nets.VariationalDropout import VariationalDropout
from nets.nets_consts import AFFINE, BATCHNORM, LAYERS_EMBEDDING, LAYERS_FULL, HIDDEN


class Layer(nn.Module):
    def __init__(self, num_in, num_out, dropout=False):
        """
        :param num_in: scalar number of input weights.
        :param num_out: scalar number of output weights.
        :param dropout: boolean for including variational dropout in each layer.
        """
        super(Layer, self).__init__()
        # initialize layer
        self.layer = nn.ModuleList([nn.Linear(num_in, num_out)])

        # batch normalization
        if BATCHNORM:
            self.layer.append(
                nn.BatchNorm1d(num_out, affine=AFFINE))

        # variational dropout
        if dropout:
            self.layer.append(VariationalDropout(num_out))

        # activation function
        self.layer.append(nn.ReLU(inplace=True))

    def forward(self, x):
        """
        :param x: tensor of shape [mbsize, N_in]
        """
        for m in self.layer:
            x = m(x)
        return x


def stack_layers(num, layers=1, dropout=False):
    """
    :param num: scalar number of input and output weights.
    :param layers: scalar number of layers to stack.
    :param dropout: boolean for including variational dropout in each layer.
    """
    # sequence of modules
    stack = nn.ModuleList([])
    for _ in range(layers):
        stack.append(Layer(num, num, dropout=dropout))
    return stack


class Embedding(nn.Module):
    def __init__(self, counts):
        """
        :param counts: dictionary of scalar input sizes.
        """
        super(Embedding, self).__init__()

        # first stack of layers: N to N
        self.layer1 = nn.ModuleDict()
        for k, v in counts.items():
            self.layer1[k] = stack_layers(v, layers=LAYERS_EMBEDDING)

        # second layer: concatenation
        num = sum(counts.values())
        self.layer2 = stack_layers(num, layers=LAYERS_EMBEDDING)

    def forward(self, x):
        """
        x: OrderedDict() with a superset of the keys in self.k
        """
        # first layer
        elements = []
        for k, mlist in self.layer1.items():
            x_k = x[k]
            for m in mlist:
                x_k = m(x_k)
            elements.append(x_k)

        # concatenate
        x = torch.cat(elements, dim=1)

        # pass through second embedding
        for m in self.layer2:
            x = m(x)

        return x


class FullyConnected(nn.Module):
    def __init__(self, num_in, num_out, dropout=True):
        """
        :param num_in: scalar number of input weights.
        :param num_out: scalar number of output parameters.
        """
        super(FullyConnected, self).__init__()
        # intermediate layer
        self.seq = nn.ModuleList([Layer(num_in, HIDDEN, dropout=dropout)])

        # fully connected network
        self.seq += stack_layers(HIDDEN, layers=LAYERS_FULL-1, dropout=dropout)

        # output layer
        self.seq += [nn.Linear(HIDDEN, num_out)]

    def forward(self, x):
        """
        :param x: tensor of shape [mbsize, num_in]
        """
        for m in self.seq:
            x = m(x)
        return x
