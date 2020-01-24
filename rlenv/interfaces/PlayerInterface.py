import numpy as np
import torch
from torch.distributions.categorical import Categorical
from rlenv.env_utils import (load_model, model_str, proper_squeeze,
                             sample_categorical, sample_bernoulli, get_con_outcomes)
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
        :return: np.array
        """
        params = self.con(sources=sources, turn=turn)
        con = self._sample_con(params=params, turn=turn)
        con_outcomes = get_con_outcomes(con=con, sources=sources, turn=turn)
        # don't draw msg if there's an acceptance or rejection
        if not self._need_msg(con):
            msg = 0.0
        else:
            msg = self._sample_msg(sources=sources, turn=turn)
        return np.append(con_outcomes, msg)

    def delay(self, sources=None, turn=0):
        input_dict = self.composer.build_input_dict(self._delay_model_name, sources=sources,
                                                    turn=turn)
        params = self.delay_model(input_dict)
        delay = sample_categorical(params)
        return delay

    def _sample_msg(self, sources=None, turn=0):
        input_dict = self.composer.build_input_dict(self._msg_model_name,sources=sources, turn=turn)
        params = self.msg_model(input_dict)
        return sample_bernoulli(params)

    def con(self, sources=None, turn=0):
        input_dict = self.composer.build_input_dict(self._con_model_name, sources=sources, turn=turn)
        params = self.con_model(input_dict)
        return params

    @staticmethod
    def _need_msg(con):
        return con != 0 and con != 1

    def _sample_con(self, params=None, turn=0):
        raise NotImplementedError()


class BuyerInterface(PlayerInterface):
    def __init__(self, composer=None):
        """
        :param composer: rlenv/Composer.Composer
        """
        super().__init__(composer=composer, byr=True)

    def _sample_con(self, params=None, turn=None):
        dist = Categorical(logits=params)
        if turn == 1:
            sample = torch.zeros(1)
            while sample == 0:
                # print('sampling concession')
                sample = dist.sample((1,))
            con = proper_squeeze(sample.float() / 100).numpy()
            # print('concession: {}'.format(con))
        elif turn == 7:
            sample = dist.sample((1,))
            while (sample > 0) and (sample < 100):
                sample = dist.sample((1,))
            con = proper_squeeze(sample.float() / 100).numpy()
        else:
            con = proper_squeeze(dist.sample((1,)).float()).numpy()
            con = con / 100
        return con


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

    def _sample_con(self, params=None, turn=None):
        con = (sample_categorical(params) / 100)
        return con
