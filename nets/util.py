import torch
import torch.nn as nn
from collections import OrderedDict
from nets.const import AFFINE, LAYERS_EMBEDDING, LAYERS_FULL, HIDDEN
from constants import MODEL_NORM


class Layer(nn.Module):
    def __init__(self, num_in, num_out, dropout=0.0, norm=MODEL_NORM):
        """
        :param num_in: scalar number of input weights.
        :param num_out: scalar number of output weights.
        :param dropout: scalar dropout rate.
        :param norm: string type of normalization.
        """
        super(Layer, self).__init__()

        # initialize module list
        self.layer = nn.ModuleList([nn.Linear(num_in, num_out)])

        # batch normalization
        if norm == 'batch':
            self.layer.append(nn.BatchNorm1d(num_out, affine=AFFINE))

        # layer normalization
        elif norm == 'layer':
            self.layer.append(nn.LayerNorm(num_out, elementwise_affine=AFFINE))

        # weight normalization
        elif norm == 'weight':
            self.layer = nn.ModuleList([nn.utils.weight_norm(self.layer[0])])

        # variational dropout
        if dropout > 0:
            self.layer.append(nn.Dropout(p=dropout, inplace=True))

        # activation function
        self.layer.append(nn.ReLU(inplace=True))

    def forward(self, x):
        """
        :param x: tensor of shape [mbsize, N_in]
        """
        for m in self.layer:
            x = m(x)
        return x


def stack_layers(num, layers=1, dropout=0.0, norm=MODEL_NORM):
    """
    :param num: scalar number of input and output weights.
    :param layers: scalar number of layers to stack.
    :param dropout: scalar dropout rate.
    :param norm: string type of normalization.
    """
    # sequence of modules
    stack = nn.ModuleList([])
    for _ in range(layers):
        stack.append(Layer(num, num, dropout=dropout, norm=norm))
    return stack


class Embedding(nn.Module):
    def __init__(self, counts, dropout=0.0, norm=MODEL_NORM):
        """
        :param counts: dictionary of scalar input sizes.
        :param dropout: scalar dropout rate.
        :param norm: string type of normalization.
        """
        super(Embedding, self).__init__()
        if len(counts) == 0:
            raise RuntimeError("Embedding must take input" +
                               "from at least 1 group")

        # first stack of layers: N to N
        self.layer1 = nn.ModuleDict()
        for k, v in counts.items():
            self.layer1[k] = stack_layers(v, layers=1, norm=norm)

        # second layer: concatenation
        num = sum(counts.values())
        self.layer2 = stack_layers(num, 
                                   layers=LAYERS_EMBEDDING,
                                   dropout=dropout, 
                                   norm=norm)

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
        x = torch.cat(elements, dim=-1)

        # pass through second embedding
        for m in self.layer2:
            x = m(x)

        return x


class FullyConnected(nn.Module):
    def __init__(self, num_in, dropout=0.0, norm=MODEL_NORM):
        """
        :param num_in: scalar number of input weights.
        :param dropout: scalar dropout rate.
        :param norm: string type of normalization.
        """
        super(FullyConnected, self).__init__()

        # intermediate layer
        self.seq = nn.ModuleList([Layer(num_in, HIDDEN, 
                                        dropout=dropout,
                                        norm=norm)])

        # fully connected network
        self.seq += stack_layers(HIDDEN,
                                 layers=LAYERS_FULL-1,
                                 dropout=dropout,
                                 norm=norm)

    def forward(self, x):
        """
        :param x: tensor of shape [mbsize, num_in]
        """
        for m in self.seq:
            x = m(x)
        return x


def create_groupings(sizes):
    groups = dict()
    groups['w2v'] = ['lstg', 'w2v_slr', 'w2v_byr']
    if 'slr' in sizes['x']:
        groups['other'] = ['lstg', 'cat', 'cndtn', 'slr']
    else:
        groups['other'] = ['lstg', 'cat', 'cndtn']
    if 'offer1' in sizes['x']:
        groups['offer'] = ['lstg'] \
            + [k for k in sizes['x'].keys() if 'offer' in k]
    return groups


def create_embedding_layers(groups=None, sizes=None,
                            dropout=0.0, norm=MODEL_NORM):
    # embeddings
    d, total = OrderedDict(), 0
    for name, group in groups.items():
        counts = {k: v for k, v in sizes['x'].items() if k in group}
        d[name] = Embedding(counts, dropout=dropout, norm=norm)
        total += sum(counts.values())
    return nn.ModuleDict(d), total
