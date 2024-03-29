import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from env.util import model_str, proper_squeeze, sample_categorical, sample_bernoulli, load_model
from constants import IDX
from featnames import CON, DELAY, MSG, SLR, BYR


class PlayerInterface:
    def __init__(self, byr=False, agent=False):
        # store args
        self.byr = byr
        self.agent = agent

    def con(self, input_dict=None, turn=None):
        params = self.query_con(input_dict=input_dict, turn=turn)
        return self.sample_con(params=params, turn=turn)

    def msg(self, input_dict=None, turn=None):
        params = self.query_msg(input_dict=input_dict, turn=turn)
        return self.sample_msg(params=params)

    def delay(self, input_dict=None, turn=None, max_interval=None):
        params = self.query_delay(input_dict=input_dict, turn=turn)
        # print('Max interval: {}'.format(max_interval))
        if max_interval is not None:
            params = params[:max_interval]
        return self.sample_delay(params=params)

    def sample_con(self, params=None, turn=None):
        raise NotImplementedError()

    def sample_msg(self, params=None, turn=None):
        raise NotImplementedError()

    def sample_delay(self, params=None, turn=None):
        raise NotImplementedError()

    def query_con(self, input_dict=None, turn=None):
        raise NotImplementedError()

    def query_msg(self, input_dict=None, turn=None):
        raise NotImplementedError()

    def query_delay(self, input_dict=None, turn=None):
        raise NotImplementedError()


class SimulatedPlayer(PlayerInterface):
    def __init__(self, byr=False, full=True):
        super().__init__(byr=byr, agent=False)

        self.full = full

        self.con_models = dict()
        self.msg_models = dict()
        self.delay_models = dict()

    def load_models(self):
        turns = IDX[BYR] if self.byr else IDX[SLR]
        for turn in turns:
            if self.full:
                self.con_models[turn] = load_model(
                    model_str(CON, turn=turn))
                if turn < 7:
                    self.msg_models[turn] = load_model(
                        model_str(MSG, turn=turn))
            if turn > 1:
                self.delay_models[turn] = load_model(
                    model_str(DELAY, turn=turn))

    def query_con(self, input_dict=None, turn=None):
        """
        :param input_dict: dict
        :param turn: current turn number
        """
        self._check_full()
        params = self.con_models[turn](input_dict).squeeze()
        return params

    def query_delay(self, input_dict=None, turn=None):
        params = self.delay_models[turn](input_dict).squeeze()
        return params

    def query_msg(self, input_dict=None, turn=None):
        self._check_full()
        params = self.msg_models[turn](input_dict).squeeze()
        return params

    def sample_msg(self, params=None, turn=None):
        return sample_bernoulli(params)

    def sample_delay(self, params=None, turn=None):
        return sample_categorical(logits=params)

    def sample_con(self, params=None, turn=None):
        raise NotImplementedError()

    @staticmethod
    def _need_msg(con):
        return con != 0 and con != 1

    def _check_full(self):
        if not self.full:
            raise NotImplementedError('Make offer cannot be called' +
                                      'since con/msg models not initalized')


class SimulatedBuyer(SimulatedPlayer):
    def __init__(self, full=True):
        super().__init__(byr=True, full=full)
        self.load_models()

    def sample_con(self, params=None, turn=None):
        if turn != 7:
            dist = Categorical(logits=params)
        else:
            dist = Bernoulli(logits=params)
        if turn == 1:
            sample = torch.zeros(1)
            while sample == 0:
                # print('sampling concession')
                sample = dist.sample((1,))
            con = proper_squeeze(sample.float() / 100).numpy()
        elif turn == 7:
            sample = dist.sample((1,))
            con = proper_squeeze(sample.float()).numpy()
        else:
            con = proper_squeeze(dist.sample((1,)).float()).numpy()
            con = con / 100
        return con


class SimulatedSeller(SimulatedPlayer):
    def __init__(self, full=True):
        """
        :param full: whether to initialize con and msg models
        (Do not if there is a Seller Agent)
        """
        super().__init__(byr=False, full=full)
        self.load_models()

    def sample_con(self, params=None, turn=None):
        con = (sample_categorical(logits=params) / 100)
        return con
