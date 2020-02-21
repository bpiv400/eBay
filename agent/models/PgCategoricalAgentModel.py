import torch
import torch.nn as nn
from torch.nn.functional import softmax
from nets.FeedForward import create_embedding_layers, create_groupings
from nets.nets_utils import FullyConnected


class PgCategoricalAgentModel(nn.Module):
    """
    Returns a vector of
    """
    def __init__(self, delay=None, sizes=None):
        """
        kwargs:
            delay boolean: gives whether model outputs a delay
            sizes: gives all sizes for the model, including size
            of each input grouping 'x' and number of elements in
            output vector 'out'
        """
        super(PgCategoricalAgentModel, self).__init__()
        self.delay = delay
        if self.delay is None or self.delay:
            raise NotImplementedError("Haven't implemented delay model")

        # expand embeddings
        groups = create_groupings(sizes=sizes)
        # create embedding layers
        self.nn0, total = create_embedding_layers(groups=groups, sizes=sizes)

        # value output layer
        self.nn1_value = FullyConnected(total, 1, dropout=False,
                                        batch_norm=False)
        # action probability output layer
        self.nn1_action = FullyConnected(total, sizes['out'], dropout=False,
                                         batch_norm=False)

    def forward(self, observation, prev_action, prev_reward):
        """
        :return: tuple of pi, v
        """
        x = observation._asdict()
        l = []
        for k in self.nn0.keys():
            l.append(self.nn0[k](x))
        embedded = torch.cat(l, dim=l[0].dim() - 1)
        v = self.nn1_value(embedded)
        logits = self.nn1_action(embedded)
        # apply softmax
        pi = softmax(logits, dim=logits.dim() - 1)
        return pi, v.squeeze()
