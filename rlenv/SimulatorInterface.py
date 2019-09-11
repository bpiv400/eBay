# TODO UPDATE DOCUMENTATION
# WRITE METHODS

import numpy as np
import torch
from torch.nn.functional import sigmoid
from rlenv.model_names import *

class SimulatorInterface:
    def __init__(self, params):
        """
        Use rl experiment params to initialize models

        :param params:
        """
        self.params = params
        self.models = {model: self._init_arrival_model(model)
                       for model in ARRIVAL_MODELS}

    def _init_arrival_model(self, model_name):
        """
        TODO: Incomplete

        Initialize an arrival model based on its name and the parameters
        of the experiment

        :param model_name: str
        :return: PyTorch Module
        """
        model = None
        return model

    def loc(self, consts=None, num_byrs=None):
        """
        Returns indicators for whether each buyer in a set of size
        num_byrs is from the US
        :param consts: tensor giving input features
        :param num_byrs: int number of buyers
        :return: 1d tensor giving whether each buyer is from the US
        """
        output = self.models[LOC].forward(consts)
        output = sigmoid(output)
        return np.random.binomial(1, output, num_byrs)

    def hist(self, consts=None, byrs_us=None):
        """
        Returns the number of previous best offer threads each
        buyer has participated in

        :param consts: tensor giving constants
        :param byrs_us: 1d np.array giving whether each byr is from the us
        :return: 1d np.array giving number of experiences each buyer has had
        """
        if np.any(byrs_us):
            # compute parameters for byrs from the us
        else:
            # compute parameters for byrs from abroad
    def num_byrs(self, consts=None):
        """
        Returns the number of buyers who arrive on a particular day

        :return:
        :param consts: constants from the lstg as tensor
        :return: int
        """
        output = self.models[NUM_BYRS].forward(consts)
        output[0] = torch.exp(output[0])
        output[1] = sigmoid(output[0])
        return np.random.negative_binomial(output[0], output[1])


    def arrival_time(self, consts=None, time_feats=None, hidden=None):
        """
        given a set of constants about a lstg
        simulate an arrival time

        Args:
            consts: np.numpy array
            time_feats: integer giving the current time
            hidden: hidden state from previous step of arrival process for this lstg. None if this is the first step
            This should be a torch.double
        Return:
            Tuple of whether a buyer arrives in the next interval and a hidden state output
        """
        raise NotImplementedError()

    def buyer_delay(self, consts=None, time_feats=None, delay=None,
                    prev_slr_offer=None, prev_byr_offer=None, hidden=None):
        """
        Given a pair of previous offers, the hidden state for the delay model, a set of constant features for
        the listing, the length of the delay thus far, and a set of time features, generate a realization for whether
        the seller makes an offer in the next interval

        Additionally update hidden state as necessary

        Args:
            hidden: torch.tensor giving the previous hidden state for the delay model
            consts: 1 dimensional np.array of constant features for the lstg
            time_feats: output of TimeFeatures.get_features (currently a dictionary) containing all time valued features
            prev_slr_offer: offer representation of last seller offer output by
            SimulatorInterface.seller_offer (currently a dictionary). None if no seller offer has occurred
            prev_byr_offer: offer representation of last buyer offer output by
            SimulatorInterface.buyer_offer (currently a dictionary)
            delay: integer giving the delay associated with this offer
        Returns:
            Tuple of an offer realization (1 if an offer is made in the next interval, 0 otherwise)
            and updated hidden state after this step. None if an offer is made
        """
        raise NotImplementedError()

    def seller_delay(self, consts=None, time_feats=None, delay=None,
                     prev_slr_offer=None, prev_byr_offer=None, hidden=None):
        """
        Given a pair of previous offers, the hidden state for the delay model, a set of constant features for
        the listing, the length of the delay thus far, and a set of time features, generate a realization for whether
        the seller makes an offer in the next interval

        Additionally update hidden state as necessary

        Args:
            hidden: torch.tensor giving the previous hidden state for the delay model
            consts: 1 dimensional np.array of constant features for the lstg
            time_feats: output of TimeFeatures.get_features (currently a dictionary) containing all time valued features
            prev_slr_offer: offer representation of last seller offer output by
            SimulatorInterface.seller_offer (currently a dictionary). None if no seller offer has occurred
            prev_byr_offer: offer representation of last buyer offer output by
            SimulatorInterface.buyer_offer (currently a dictionary)
            delay: integer giving the delay associated with this offer
        Returns:
            Tuple of an offer realization (1 if an offer is made in the next interval, 0 otherwise)
            and updated hidden state after this step. None if an offer is made
        :return:
        """
        raise NotImplementedError()

    def buyer_offer(self, consts=None, hidden=None, time_feats=None, prev_slr_offer=None, prev_byr_offer=None,
                    prev_slr_delay=None, prev_byr_delay=None, delay=None):
        """
        Given a previous buyer offer, a previous seller offer, a dictionary of previous hidden states,
        a set of time valued features, and the amount of delay chosen for this offer, simulate the next buyer offer

        Also updates the relevant hidden states
        Args:
            hidden: dictionary of tensors giving previous hidden states
            of the buyer simulator models ('concession', 'delay', 'round')
            consts: 1 dimensional np.array of constant features for the lstg
            time_feats: output of TimeFeatures.get_features (currently a dictionary) containing all time valued features
            prev_slr_offer: offer representation of last seller offer output by
            SimulatorInterface.seller_offer (currently a dictionary). None if no seller offer has occurred
            prev_byr_offer: offer representation of last buyer offer output by
            SimulatorInterface.buyer_offer (currently a dictionary)
            prev_slr_delay: integer giving the number of seconds the seller delayed for their previous offer
            prev_byr_delay: integer giving the number of seconds the buyer delayed for their previous offer
            delay: integer giving the delay associated with this offer
        Returns:
            Tuple of an offer representation (currently a dictionary that is expected to contain concession [0, 1] and
            price (real price of the offer)--nothing else is assumed. Should contain all other features used
            as inputs later AND hidden state dictionary (should have delay = None, and concession/round from this
            most recent step
        """
        raise NotImplementedError()

    def slr_offer(self, consts=None, hidden=None, time_feats=None, prev_slr_offer=None, prev_byr_offer=None,
                  prev_slr_delay=None, prev_byr_delay=None, delay=None):
        """
        Given a previous buyer offer, a previous seller offer, a dicitonary of previous hidden states for the
        seller models, a set of time valued features,
        and the amount of delay chosen for this offer, simulate the next seller offer

        Also updates the relevant hidden states
        Args:
            hidden: dictionary of tensors giving previous hidden states
            of the buyer simulator models ('concession', 'delay', 'round')
            consts: 1 dimensional np.array of constant features for the lstg
            time_feats: output of TimeFeatures.get_features (currently a dictionary) containing all time valued features
            prev_slr_offer: offer representation of last seller offer output by
            SimulatorInterface.seller_offer (currently a dictionary). None if no seller offer has occurred
            prev_byr_offer: offer representation of last buyer offer output by
            SimulatorInterface.buyer_offer (currently a dictionary)
            prev_slr_delay: integer giving the number of seconds the seller delayed for their previous offer
            prev_byr_delay: integer giving the number of seconds the buyer delayed for their previous offer
            delay: integer giving the delay associated with this offer
        Returns:
            Tuple of an offer representation (currently a dictionary that is expected to contain concession [0, 1] and
            price (real price of the offer)--nothing else is assumed. Should contain all other features used
            as inputs later AND hidden state dictionary (should have delay = None, and concession/round from this
            most recent step
            """
        raise NotImplementedError()
