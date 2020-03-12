import torch
from train.Trainer import Trainer


class EntropyTrainer(Trainer):
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
        super().__init__(name, train_part, test_part, dev, device)

    @staticmethod
    def _get_entropy(lnq):
        h = -torch.sum(torch.exp(lnq) * lnq, dim=-1)
        return h

    @staticmethod
    def _get_kl(lnq, p):
        kl = p * (torch.log(p) - lnq)
        kl[p == 0] = 0
        kl = torch.sum(kl, dim=-1)
        return kl

    def _get_penalty(self, lnq, b):
        if self.is_delay:
            return self._get_entropy(lnq)
        return self._get_kl(lnq, b['p'])
