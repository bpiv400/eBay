from models.nets import FeedForward
from constants import *


# constructs a neural network to discriminate between real and fake threads
class Disciminator:
    def __init__(self, sizes, dropout=True, device='cuda'):
        '''
        sizes: dictionary of data sizes
        gamma: scalar coefficient on regularization term
        device: either 'cuda' or 'cpu'
        '''

        # save parameters from inputs
        self.device = device
        self.dropout = dropout

        # initialize gamma to 0
        self.gamma = 0.0

        # feed-forward neural net
        self.net = FeedForward(sizes, dropout=dropout).to(device)


    def set_gamma(self, gamma):
        if gamma > 0:
            if not self.dropout:
                error('Gamma cannot be non-zero without dropout layers.')
            self.gamma = gamma


    def get_penalty(self, factor=1):
        penalty = torch.tensor(0.0, device=self.device)
        for m in self.net.modules():
            if hasattr(m, 'kl_reg'):
                penalty += m.kl_reg()
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


    def run_batch(self, d, factor, optimizer=None):
        # train / eval mode
        isTraining = optimizer is not None
        self.net.train(isTraining)

        # predicted probability of fake
        theta = torch.exp(self.net(d).squeeze())
        pfake = torch.sigmoid(theta)

        # loss: 1 if guess is wrong
        loss = torch.sum(d['y'] != (pfake > 0.5)).float()

        # add in KL divergence and step down gradients
        if isTraining:
            # add in regularization penalty
            if self.gamma > 0:
                loss = loss + self.get_penalty(factor)

            # step down gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # return log-likelihood
        return loss.item()

