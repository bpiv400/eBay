import torch, torch.nn as nn

MAX_SEQ_LEN = 3
N_OUTPUT = 5

class Model(nn.Module):
    def __init__(self, N_fixed, N_offer, N_hidden, N_output, dropout):
        # super constructor
        super(Model, self).__init__()

        # initial hidden nodes
        self.h0 = nn.Linear(N_fixed, N_hidden)

        # initial cell nodes
        self.c0 = nn.Linear(N_fixed, N_hidden)

        # activation function
        self.f = nn.Sigmoid()

        # lstm layer
        self.lstm = nn.LSTM(input_size=N_offer, hidden_size=N_hidden,
            bias=True, dropout=dropout)

        # output layer
        self.output = nn.Linear(N_hidden, N_OUTPUT)


    @staticmethod
    def expectation(x):
        '''
        Takes expectation of predicted distribution.

        Inputs x: parameters (seq_len, batch_size, num_param)
        Outputs: expectation (seq_len, batch_size)
        '''
        gamma0 = torch.exp(x[:,:,1])
        gamma1 = torch.exp(x[:,:,2])
        gamma12 = torch.exp(x[:,:,3])
        alpha = 1 + torch.exp(x[:,:,4])
        beta = 1 + torch.exp(x[:,:,5])

        den = 1 + gamma0 + gamma1 + gamma12
        num = gamma1 + torch.div(gamma12, 2) + torch.div(alpha, alpha+beta)

        return torch.div(num, den)


    def forward(self, x_fixed, x_offer):
        # calculate initial hidden layer and cell state and process offer data
        x, _ = self.lstm(x_offer, (self.f(self.h0(x_fixed)), self.f(self.c0(x_fixed))))

        # ensure that predictions are padded to MAX_SEQ_LEN
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=MAX_SEQ_LEN)

        # output layer: (seq_len, batch_size, num_param)
        x = self.output(x)

        # take expectation
        return expectation(x)