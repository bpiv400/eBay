from simulator.nets import *
from constants import *


# constructs a neural network to discriminate between real and fake threads
class Disciminator:

    def __init__(self, sizes, gamma=0.0, device='cuda'):
        '''
        sizes: dictionary of data sizes
        gamma: scalar coefficient on regularization term
        device: either 'cuda' or 'cpu'
        '''

        # save parameters from inputs
        self.gamma = gamma
        self.device = device

        # subsequent neural net(s)
        self.net = FeedForward(sizes, dropout=gamma > 0).to(device)


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


    def run_batch(self, d0, d1, factor, optimizer=None):
        # train / eval mode
        isTraining = optimizer is not None
        self.net.train(isTraining)

        # predictions from model
        t0 = torch.exp(self.net(d0).squeeze())
        t1 = torch.exp(self.net(d1).squeeze())

        # loss: predicted probability of fake
        loss = torch.sum(t1 / (t0 + t1))

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

