"""
actor.py

Script containing classes of rl actor models

Classes:
    DeterministicActor: Actor which takes deterministic
    action with respect to a given history of states
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Simulator


class DeterministicActor(nn.Module):
    """
    Actor which takes deterministic
    action with respect to a given history of states

    Functions:

    """
    MAX_SEQ_LEN = 4

    def __init__(self, n_fixed, n_hidden, n_offer,
                 n_hidden_layers):
        # initial hidden nodes
        self.h0 = nn.Linear(n_fixed, n_hidden)

        # initial cell nodes
        self.c0 = nn.Linear(n_fixed, n_hidden)

        # lstm layer
        self.lstm = nn.LSTM(input_size=n_offer, hidden_size=n_hidden,
                            bias=True, num_layers=n_hidden_layers, nonlinearity="relu")

        # output layer for softmaxing over accept, reject, 50% counter offer no message,
        # 50 % counter offer with message, other counter with message,
        #  and other counter offer without message
        self.basic_out = nn.Linear(in_features=n_hidden, out_features=6)
        self.final_out = nn.Linear(in_features=n_hidden, out_features=1)
        self.final_act = nn.Sigmoid()
        # output layer for time for 50% offer (input=hidden state concatenated
        # indicator for message with for message)
        # hidden, 1, 0 = message
        # hidden, 0, 1 = no message
        self.time_layer_50 = nn.Linear(
            in_features=n_hidden + 1, out_features=2)

        # offer output layer for non-50% offer (input = hidden state concatenated
        # with indicator for message)
        self.offer_layer = nn.Linear(in_features=n_hidden + 1, out_features=2)

        # time output layer for non-50% offer (input = hidden state concatenated
        # with input for message and value of counter)
        # hidden, 1, 0, offer = message
        # hidden 0, 1, offer = no message
        self.time_counter = nn.Linear(in_features=n_hidden + 2, out_features=2)

    def step(self, x):
        """
        Executes exactly one actor step, assuming hidden state and
        cell have already been initialized

        Assumes input has not been packed (no reason since only 1 step)
        """
        x = torch.unsqueeze(
            x, 0)  # add singleton dimension denoting number of steps

        # initialize first hidden and cell states
        _, (self.h1, self.c1) = self.lstm(x, (self.h1, self.c1))
        actions, log_probs = self.action_from_state()
        return actions, log_probs

    def final_step(self, x):
        """
        Executes the final bargaining step, where the buyer must choose
        between accepting and rejecting the seller's last offer

        Assumes input has not been packed
        """
        x = torch.unsqueeze(
            x, 0)  # add singleton dimension denoting number of steps

        # initialize first hidden and cell states
        _, (self.h1, self.c1) = self.lstm(x, (self.h1, self.c1))
        y = torch.squeeze(self.h1)
        y = self.final_out(y)
        y = self.final_act(y)
        return y

    @staticmethod
    def populate_sample_message_select(h1, first_inds, sec_inds, offr=None):
        # create tensor of dim (num mes_50 + num nomes_50, hidden_size + 2)
        if offr is None:
            out = torch.autograd.Variable(torch.zeros(first_inds.size() +
                                                      sec_inds.size(), h1.size()[2] + 1))
        else:
            out = torch.autograd.Variable(torch.zeros(first_inds.size() +
                                                      sec_inds.size(), h1.size()[2] + 1))

        # let the first rows denote the threads where a message is sent
        # (leaving the remaining rows for nomes)
        # let the index after the hidden state be an indicator for sending a message
        out[:first_inds.size(), h1.size()[2]] = 1

        # if this is the time layer input for the counter offer, add the generated offers
        if offr is not None:
            out[:, h1.size()[2] + 1] = offr

        # set the first n_hidden dimensions of each row to the corresponding hidden states
        # for the message inds then the nomessage inds
        out[:first_inds.size(), :h1.size()[2]] = h1[1, first_inds, :]
        out[first_inds.size():, :h1.size()[2]] = h1[1, sec_inds, :]

        return out

    def action_from_state(self):
        """
        Add documentation

        Conducted experiment to ensure gradients are tracked across slices when
        torch.zeros is initialized as a variable before slicing, so this code should
        not lose gradients

        Action summary
        0 - indicator for whether a message is given (nan if accept/reject)
        1 - indicator for whether offer is 50 / 50 (if any) OR
        1 - (IF ACCEPT/REJECT) indicator for accept
        2 - normalized value of offer (if any)
        3 - normalized buyer delay

        CONCERN: differentiation through slice copies is slow w
        """
        # remove first single dimension
        h1 = torch.squeeze(self.h1)
        x = self.basic_out(h1)
        x = torch.nn.soft_max(x, 0)
        # sample from softmax layer
        basic_choice = torch.multinomial(x, 1)
        # remove singleton dimension
        basic_choice = torch.squeeze(basic_choice)

        # let 0 correspond to accept
        accept_inds = (basic_choice == 0).nonzero()
        # 1 correspond to reject
        rej_inds = (basic_choice == 1).nonzero()
        # let 2 correspond to 50% with message
        mes_50_inds = (basic_choice == 2).nonzero()
        # let 3 correspond to 50% with no message
        nomes_50_inds = (basic_choice == 3).nonzero()
        # let 4 correspond to non- 50% with message
        mes_counter_inds = (basic_choice == 4).nonzero()
        # let 5 correspond to non-50% with no message
        nomes_counter_inds = (basic_choice == 5).nonzero()

        # create tensor of dim (num mes_50 + num nomes_50, hidden_size + 2)
        time_50_in = self.populate_sample_message_select(
            h1, mes_50_inds, nomes_50_inds)
        # generate alpha, beta of the beta distribution
        time_50_out = self.time_50_out(time_50_in)
        # offer_in initialization
        offer_in = self.populate_sample_message_select(
            h1, mes_counter_inds, nomes_counter_inds)

        # compute alpha beta output values
        offr_out = self.offr_layer(offer_in)
        # sample from beta distribution
        offrs_dist = torch.distributions.beta.Beta(
            offr_out[:, 0], offr_out[:, 1])
        offrs = offrs_dist.sample()
        # compute input to time counter layer
        time_counter_in = self.populate_sample_message_select(h1, mes_counter_inds,
                                                              nomes_counter_inds, offr=offrs)
        # generate alpha, beta for counter time distribution
        time_counter_out = self.time_counter(time_counter_in)
        # create beta distributions for counter offer times
        time_counter_dist = torch.distributions.beta.Beta(
            time_counter_out[:, 0], time_counter_out[:, 1])
        # sample from it
        times_counter = time_counter_dist.sample()
        # now, do the same for 50% offer times
        time_50_dist = torch.distributions.beta.Beta(
            time_50_out[:, 0], time_50_out[:, 1])
        times_50 = time_50_dist.sample()

        # initialize outputs
        # actions[:, 0] gives a message indicator
        # actions[:, 1] gives a 50/50 indicator
        # actions[;, 2] gives a real valued counter offer value
        # actions[:, 3] gives a time valued indicator
        #################################
        # if accept or reject,
        # actions[:, 0] = tensor.NaN
        # actions[:, 1] = 0 => reject
        # actions[:, 1] = 1 => accept
        actions = torch.zeros(h1.size()[1], 5)
        log_probs = torch.autograd.Variable(torch.zeros(h1.size()[1]))

        # calculate log probabilities for accept and reject
        log_probs[accept_inds] = torch.log(x[accept_inds])
        log_probs[rej_inds] = torch.log(x[rej_inds])

        # calculate log probabilities for split-difference offers
        inds_50 = torch.cat((mes_50_inds, nomes_50_inds))
        log_probs[inds_50] = torch.log(
            x[inds_50]) + time_50_dist.log_prob(times_50)

        # calculate log probabilities for counter offs
        counter_inds = torch.cat((mes_counter_inds, nomes_counter_inds))
        log_probs[counter_inds] = torch.log(x[counter_inds]) + time_counter_dist.log_prob(times_counter) +\
            offrs_dist.log_prob(offrs)

        # action summary
        # 0 - indicator for whether a message is given (nan if accept/reject)
        # 1 - indicator for whether offer is 50 / 50 (if any) OR
        # 1 - (IF ACCEPT/REJECT) indicator for accept
        # 2 - normalized value of offer (if any)
        # 3 - normalized buyer delay
        # calculate actions message indicator
        actions[mes_50_inds, 0] = 1
        actions[mes_counter_inds, 0] = 1
        # calculate accept/reject indicators
        actions[accept_inds, 0] = float('nan')
        actions[accept_inds, 1] = 1
        actions[rej_inds, 0] = float('nan')
        # calculate 50/50 indicator
        actions[nomes_50_inds, 1] = 1
        actions[mes_50_inds, 1] = 1
        # calculate offer value
        actions[inds_50, 2] = .5
        actions[counter_inds, 2] = offrs
        # calculate time values
        actions[inds_50, 3] = times_50
        actions[counter_inds, 3] = times_counter
        # finally shrink hidden state as necessary
        self.shrink_hidden(torch.cat((inds_50, counter_inds)))
        return actions, log_probs

    def init_hidden_state(self, h0, c0):
        # assumes input has already been transformed to correct hidden state and cell sizes
        # (1, batch_size, hidden_size)
        # replicates to number of layers
        h0 = self.h0(h0)
        c0 = self.c0(c0)
        self.h1 = Simulator.increase_num_layers(h0, self.num_layers, False)
        self.c1 = Simulator.increase_num_layers(c0, self.num_layers, False)

    def shrink_hidden(self, keep_inds=None):
        # removes examples where the actor chose to accept
        # the most recent seller offer or reject the seller
        # offer altogether
        # when called after the first and second buyer offers have been generated,
        # also removes examples where the seller chose to accept the buyer
        # offer.
        # When called after the third buyer offer, removes the threads
        # where the last seller offer is a rejection in addition to
        # all other removals
        self.h1 = self.h1[:, keep_inds, :]
        self.c1 = self.c1[:, keep_inds, :]

    # DEPRECATED
    # deprecated just use get_next_offr
    # def get_first_offr(self, consts):
    #    # init_hidden = self.h0(consts)
    #    # init_hidden = self.f(init_hidden)
    #    # init_cell = self.c0(consts)
    #    # init_cell = self.f(init_cell)
    #    # assumes consts have already been transformed
    #
    #    n_seq_feats = self.lstm.input_size
    #    batch_size = consts.shape[1]
    #    zeros = torch.zeros([1, batch_size, n_seq_feats],
    #                        dtype=torch.float64)
    #
    #    _, next_state = self.lstm(zeros, (init_hidden, init_cell))
    #    self.h1, self.c1 = next_state
    #    return self.action_from_state(self.h1)
