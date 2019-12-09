from simulator.nets import *
from constants import *


# constructs model-specific neural network.
class Simulator:

    def __init__(self, model, sizes, dropout, device='cuda'):
        '''
        model: one of 'arrival', 'hist', 'delay_byr', 'delay_slr',
            'con_byr', 'con_slr'
        sizes: dictionary of data sizes
        dropout: dropout rates for [embedding, fully-connected]
        device: either 'cuda' or 'cpu'
        '''

        # save parameters from inputs
        self.model = model
        self.device = device

        # sloss function
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
            self.net = LSTM(sizes, dropout).to(device)
        else:
            self.net = FeedForward(sizes, dropout).to(device) 


    def run_batch(self, d, optimizer=None):
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

        # step down gradients
        if isTraining:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # return log-likelihood
        return -loss.item()

