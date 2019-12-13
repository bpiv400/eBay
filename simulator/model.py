from simulator.nets import *
from constants import *


# constructs model-specific neural network.
class Simulator:

    def __init__(self, model, sizes, dropout=True, device='cuda'):
        '''
        model: one of 'arrival', 'hist', 'delay_byr', 'delay_slr',
            'con_byr', 'con_slr'
        sizes: dictionary of data sizes
        gamma: scalar coefficient on regularization term
        device: either 'cuda' or 'cpu'
        '''

        # save parameters from inputs
        self.model = model
        self.device = device
        self.dropout = dropout

        # initialize gamma to 0
        self.gamma = 0.0

        # loss function
        if model in ['hist', 'con_slr']:
            self.loss = torch.nn.CrossEntropyLoss(
                reduction='sum')
        elif model == 'con_byr':
            self.loss = [torch.nn.CrossEntropyLoss(reduction='sum'),
                         torch.nn.BCEWithLogitsLoss(reduction='sum')]
        elif model == 'arrival':
            self.loss = torch.nn.PoissonNLLLoss(
                log_input=True, reduction='sum')
        else:
            self.loss = torch.nn.BCEWithLogitsLoss(
                reduction='sum')

        # subsequent neural net(s)
        if ('delay' in model) or (model == 'arrival'):
            self.net = LSTM(sizes, dropout=dropout).to(device)
        else:
            self.net = FeedForward(sizes, dropout=dropout).to(device)


    def set_gamma(self, gamma):
        if (gamma > 0) and not self.dropout:
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

        # prediction from model and loss for recurrent models
        if 'x_time' in d:
            theta = self.net(d['x'], d['x_time'])
            mask = d['y'] > -1
            loss = self.loss(theta[mask], d['y'][mask])
        else:
            theta = self.net(d['x']).squeeze()

            if self.model == 'con_byr':
                # observation is on buyer's 4th turn if all three turn indicators are 0
                t4 = torch.sum(d['x']['offer1'][:,-3:], dim=1) == 0

                # loss for first 3 turns
                loss = self.loss[0](theta[~t4,:], d['y'][~t4].long())

                # loss for last turn: use accept probability only
                if torch.sum(t4).item() > 0:
                    loss += self.loss[1](theta[t4,-1], 
                        (d['y'][t4] == 100).float())

            else:
                loss = self.loss(theta, 
                    d['y'].float() if 'msg' in self.model else d['y'])

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

