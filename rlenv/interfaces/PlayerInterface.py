import pandas as pd
import torch
from torch.distributions.categorical import Categorical
from rlenv.env_utils import (load_model, model_str, featname, proper_squeeze,
                             sample_categorical, sample_bernoulli,
                             update_byr_outcomes, update_slr_outcomes)
from rlenv.env_consts import BYR_PREFIX, SLR_PREFIX
from featnames import *


class PlayerInterface:
    def __init__(self, composer=None, byr=False):
        # store args
        self.composer = composer
        self.byr = byr

        # store names for each
        self._con_model_name = model_str(CON, byr=byr)
        self._msg_model_name = model_str(MSG, byr=byr)
        self._delay_model_name = model_str(DELAY, byr=byr)

        self.msg_model = load_model(self._msg_model_name)
        self.con_model = load_model(self._con_model_name)
        self.delay_model = load_model(self._delay_model_name)

    def make_offer(self, sources=None, turn=None):
        """
        Returns updated turn outcome pd.Series with result of a message
        and delay sampled from relevant models
        :param sources: Sources
        :param turn: current turn number
        :return: pd.Series
        """
        params = self.con(sources=sources)
        outcomes, sample_msg = self._sample_con(params=params, sources=sources, turn=turn)
        # don't draw msg if there's an acceptance or rejection
        if not sample_msg:
            return outcomes
        else:
            self._sample_msg(sources=sources, outcomes=outcomes, turn=turn)
        return outcomes

    def delay(self, sources=None):
        input_dict = self.composer.build_input_dict(self._delay_model_name, sources=sources)
        params = self.delay_model(input_dict)
        delay = sample_categorical(params)
        return delay

    def _sample_msg(self, sources=None, outcomes=None, turn=0):
        input_dict = self.composer.build_input_dict(self._msg_model_name,sources=sources)
        params = self.msg_model(input_dict)
        outcomes[featname(MSG, turn)] = sample_bernoulli(params)

    def _sample_con(self, params=None, sources=None, turn=0):
        raise NotImplementedError()

    def con(self, sources=None):
        input_dict = self.composer.build_input_dict(self._con_model_name, sources=sources)
        params = self.con_model(input_dict)
        return params


class BuyerInterface(PlayerInterface):
    def __init__(self, composer=None):
        """
        :param composer: rlenv/Composer.Composer
        """
        super().__init__(composer=composer, byr=True)

    def _sample_con(self, params=None, turn=None, sources=None):
        # resample until non reject on turn 1
        outcomes = pd.Series(0.0, index=ALL_OUTCOMES[turn])
        dist = Categorical(logits=params)
        if turn == 1:
            sample = torch.zeros(1)
            while sample == 0:
                # print('sampling concession')
                sample = dist.sample((1,))
            con = proper_squeeze(sample.float() / 100).numpy()
            # print('concession: {}'.format(con))
        else:
            con = proper_squeeze(dist.sample((1,)).float()).numpy()
            con = con / 100
            if turn == 7 and con != 1:  # On the last turn, recast all buckets to 0 except accept
                con = 0
        outcomes, sample_msg = update_byr_outcomes(con=con, sources=sources, turn=turn)
        return outcomes, sample_msg


class SellerInterface(PlayerInterface):
    def __init__(self, composer=None, full=True):
        """
        :param composer: rlenv/Composer.Composer
        :param full: whether to initialize con and msg models
        (Do not if there is a Seller Agent)
        """
        super().__init__(composer=composer, byr=False)
        self.full = full
        # throw out con and msg models if there's an agent
        if not full:
            self.con_model = None
            self.msg_model = None

    def make_offer(self, sources=None, turn=None):
        """
        Returns updated turn outcome pd.Series with result of a message
        and delay sampled from relevant models
        :param sources: Sources
        :param turn: current turn number
        :return: pd.Series
        """
        if not self.full:
            raise NotImplementedError('Make offer cannot be called' +
                                      'since con/msg models not initalized')
        return super().make_offer(sources=sources, turn=turn)

    def _sample_con(self, params=None, turn=None, sources=None):
        con = (sample_categorical(params) / 100)
        # break out into separate util function
        outcomes, sample_msg = update_slr_outcomes(con=con, sources=sources, turn=turn)
        return outcomes, sample_msg
