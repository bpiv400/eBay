import torch, torch.nn as nn
from model.nets import FeedForward


class Model:
    def __init__(self, name, sizes, params, device='cuda'):
        '''
        Creates a neural net and manages training and validation.
        :param name: string name of model.
        :param sizes: dictionary of data sizes.
        :param params: dictionary of neural net parameters.
        :param device: either 'cuda' or 'cpu'
        '''
        # initialize regularization terms to 0
        self.gamma = 0.0
        self.smoothing = 0.0

        # loss function
        if name in ['hist', 'con_slr', 'con_byr']:
            self.loss = nn.CrossEntropyLoss(reduction='sum')
        elif 'msg' in name:
            self.loss = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            self.loss = self._TimeLoss

        # neural net
        self.net = FeedForward(sizes, params).to(device)


    def get_penalty(self, factor=1):
        penalty = 0.0
        for m in self.net.modules():
            if hasattr(m, 'kl_reg'):
                penalty += m.kl_reg().item()
        return self.gamma * factor * penalty


    def get_alpha_stats(self, threshold=9):
        above, total, largest = 0.0, 0.0, 0.0
        for m in self.net.modules():
            if hasattr(m, 'log_alpha'):
                alpha = torch.exp(m.log_alpha)
                largest = max(largest, torch.max(alpha).item())
                total += alpha.size()[0]
                above += torch.sum(alpha > threshold).item()
        return above / total, largest
