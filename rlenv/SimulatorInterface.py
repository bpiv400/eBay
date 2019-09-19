# TODO UPDATE DOCUMENTATION
# WRITE METHODS

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.negative_binomial import NegativeBinomial
import pandas as pd
import utils
from torch.nn.functional import sigmoid
from rlenv.model_names import *
from rlenv.Composer import Composer
from rlenv import env_consts
from simulator.nets import FeedForward, RNN, LSTM


class SimulatorInterface:
    def __init__(self, params):
        """
        Use rl experiment params to initialize models

        :param params:
        """
        self.models = dict()
        self.params = params
        for model in MODELS:
            self.models[model] = self._init_model(model, params[model])
        self.composer = Composer(params)

    @staticmethod
    def _init_model(model_name, model_exp):
        """
        Initialize pytorch network for some model

        :param model_name: str
        :param model_type: str
        :param model_exp: experiment number for the model
        :return: PyTorch Module
        """
        if SLR_PREFIX in model_name:
            model_type = SLR_PREFIX
        elif BYR_PREFIX in model_name:
            model_type = BYR_PREFIX
        else:
            model_type = ARRIVAL_PREFIX
        model_dir = '{}/{}/{}/'.format(env_consts.MODEL_DIR,
                                       model_type, model_name)
        model_params_path = '{}params.csv'.format(model_dir)
        sizes_path = '{}sizes.pkl'.format(model_dir)
        model_path = '{}{}.pt'.format(model_dir, model_exp)

        sizes = utils.unpickle(sizes_path)
        params = pd.from_csv(model_params_path, index_col='id')
        params = params.loc[model_exp].to_dict()
        if model in FEED_FORWARD:
            net = FeedForward(params, sizes, toRNN=False)
        elif model in LSTM:
            net = LSTM(params, sizes)
        else:
            net = RNN(params, sizes)
        net.load_state_dict(torch.load(model_path))
        return net

    @staticmethod
    def _bernoulli_sample(logits, num_byrs, ff=True):
        """

        :param logits:
        :param num_byrs:
        :param ff:
        :return:
        """
        dist = Bernoulli(logits=logits)
        return dist.sample((num_byrs, )).squeeze()

    def loc(self, consts=None, clock_feats=None, num_byrs=None):
        """
        Returns indicators for whether each buyer in a set of size
        num_byrs is from the US
        :param consts: tensor giving input features
        :param clock_feats:
        :param num_byrs: int number of buyers
        :return: 1d tensor giving whether each buyer is from the US
        """
        x_fixed = self.composer.loc_input(consts=consts, clock_feats=clock_feats)
        output = self.models[LOC].simulate(x_fixed)
        locs = SimulatorInterface._bernoulli_sample(output, num_byrs, ff=True)
        return locs

    def hist(self, consts=None, clock_feats=None, byr_us=None):
        """
        Returns the number of previous best offer threads each
        buyer has participated in

        :param consts: 1-dimensional np.array giving constants of the lstg
        :param clock_feats: 1-dimensional np.array giving clock features
        :param byr_us: 1d np.array giving whether each byr is from the us
        :return: 1d np.array giving number of experiences each buyer has had
        """
        us_count = np.count_nonzero(byr_us)
        foreign = us_count < len(byr_us)
        us = us_count > 0
        x_fixed = self.composer.hist_input(consts=consts, clock_feats=clock_feats,
                                           us=us, foreign=foreign)
        output = self.models[HIST].simulate(x_fixed)

        params = torch.zeros(len(byr_us), 2)
        if foreign and us:
            foreign = byr_us == 0
            params[foreign, :] = output[0, :]
            params[~foreign, :] = output[1, :]
        else:
            params[:, :] = output[0, :]
        hists = SimulatorInterface._sample_negative_binomial(params)
        return hists

    @staticmethod
    def _sample_negative_binomial(params):
        """
        Returns a sample from a batched negative binomial distribution

        :param params:
        :return:
        """
        dist = NegativeBinomial(total_count=torch.exp(params[:, 0]), logits=params[:, 1])
        sample = dist.sample((1, ))
        return sample.squeeze()

    def days(self, consts=None, clock_feats=None, hidden=None):
        """
        Returns the number of buyers who arrive on a particular day

        :param consts: constants from the lstg as 1 dimensional np array
        :param hidden: tuple giving hidden state of days model
        :param clock_feats: 1d np.array containing clock features
        :return: tuple of number of buyers, hidden state
        """
        need_fixed = hidden is None
        x_fixed, x_time = self.composer.days_input(consts=consts,
                                                   clock_feats=clock_feats,
                                                   fixed=need_fixed)
        params, hidden = self.models[DAYS].simulate(x_time, x_fixed=x_fixed,
                                                    hidden=hidden)
        params = params.squeeze().unsqueeze(-1)
        sample = SimulatorInterface._sample_negative_binomial(params)
        return sample

    def sec(self, consts=None, clock_feats=None, byr_us=None, byr_hist=None):
        """
        Returns the time of day each byr arrives

        :param consts: 1-dimensional np.array giving constants of the lstg
        :param clock_feats: 1-dimensional np.array giving clock features
        :param byr_us: 1d np.array giving whether each byr is from the us
        :param byr_hist: 1d np.array giving how much experience each byr has
        :return: 1d np.array giving number of experiences each buyer has had
        """
        x_fixed = self.composer.sec_input(consts=consts,
                                          clock_feats=clock_feats,
                                          byr_us=byr_us,
                                          byr_hist=byr_hist)
        output = self.models[SEC].simulate(x_fixed)

        output = output.reshape(len(byr_us), 3, -1).permute(0, 2, 1)
        output[:, :, [0, 1]] = torch.exp(output[:, :, [0, 1]]) + 1
        ancestor = Categorical(logits=output[:, :, 2])
        draws = ancestor.sample(sample_shape=(1, ))
        beta_params = output[torch.arange(output.shape[0]), draws[0, :], :]
        beta = Beta(beta_params[:, 0], beta_params[:, 1])
        secs = beta.sample((1, ))
        return secs[1, :]

    def bin(self, consts=None, clock_feats=None, byr_us=None, byr_hist=None):
        """
        Returns whether each buyer chooses to buy the listing now

        :param consts:
        :param clock_feats:
        :param byr_us:
        :param byr_hist:
        :return:
        """
        x_fixed = self.composer.bin_input(consts=consts,
                                          clock_feats=clock_feats,
                                          byr_us=byr_us,
                                          byr_hist=byr_hist)
        output = self.models[BIN].simulate(x_fixed)
        bins = SimulatorInterface._bernoulli_sample(output, len(byr_us), ff=True)
        return bins




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
