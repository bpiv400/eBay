import torch
from torch.distributions.categorical import Categorical
from rlenv.env_utils import (model_str, proper_squeeze,
                             sample_categorical, sample_bernoulli)
from featnames import *
from utils import load_model


class PlayerInterface:
    def __init__(self, byr=False):
        # store args
        self.byr = byr

        self.con_models = dict()
        self.msg_models = dict()
        self.delay_models = dict()

        self.load_models()

    def load_models(self):
        if self.byr:
            turns = [1, 3, 5, 7]
        else:
            turns = [2, 4, 6]
        for turn in turns:
            self.delay_models[turn] = load_model(model_str(DELAY, turn=turn))
            self.con_models[turn] = load_model(model_str(CON, turn=turn))
            if turn != 7:
                self.msg_models[turn] = load_model(model_str(MSG, turn=turn))

    def con(self, input_dict=None, turn=None):
        """
        :param input_dict: dict
        :param turn: current turn number
        :return: np.float
        """
        params = self.con_models[turn](input_dict).squeeze()
        con = self.sample_con(params=params, turn=turn)
        return con

    def delay(self, input_dict=None, turn=None):
        params = self.delay_models[turn](input_dict).squeeze()
        delay = sample_categorical(params)
        return delay

    def msg(self, input_dict=None, turn=None):
        params = self.msg_models[turn](input_dict).squeeze()
        return sample_bernoulli(params)

    @staticmethod
    def _need_msg(con):
        return con != 0 and con != 1

    def sample_con(self, params=None, turn=0):
        raise NotImplementedError()


class BuyerInterface(PlayerInterface):
    def __init__(self):
        super().__init__(byr=True)

    def sample_con(self, params=None, turn=None):
        dist = Categorical(logits=params)
        if turn == 1:
            sample = torch.zeros(1)
            while sample == 0:
                # print('sampling concession')
                sample = dist.sample((1,))
            con = proper_squeeze(sample.float() / 100).numpy()
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
    def __init__(self, full=True):
        """
        :param full: whether to initialize con and msg models
        (Do not if there is a Seller Agent)
        """
        super().__init__(byr=False)
        self.full = full
        # throw out con and msg models if there's an agent
        if not full:
            self.con_models = None
            self.msg_models = None

    def con(self, input_dict=None, turn=None):
        """
        Generate a concession if concession model defined
        :param input_dict: dict
        :param turn: current turn number
        :return: np.float
        """
        self._check_full()
        return super().con(input_dict=input_dict, turn=turn)

    def msg(self, input_dict=None, turn=None):
        """
        Generate a msg if msg model defined
        :return: np.float
        """
        self._check_full()
        return super().msg(input_dict=input_dict, turn=turn)

    def sample_con(self, params=None, turn=None):
        con = (sample_categorical(params) / 100)
        return con

    def _check_full(self):
        if not self.full:
            raise NotImplementedError('Make offer cannot be called' +
                                      'since con/msg models not initalized')
