from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch
from rlenv.env_utils import proper_squeeze, categorical_sample, get_split
from rlenv.composer.maps import O_OUTCOMES_MAP, L_OUTCOMES_MAP
from rlenv.env_consts import NORM_POS


class SimulatedActor:
    def __init__(self, model=None):
        self.delay_hidden = None
        self.con_hidden = None
        self.msg_hidden = None
        self.model = model

    def make_offer(self, sources=None, turn=1):
        params, self.con_hidden = self.model.con(sources=sources, hidden=self.con_hidden, turn=turn)
        outcomes = self.sample_con(params, sources=sources, turn=turn)
        params, self.msg_hidden = self.model.msg(sources=sources, hidden=self.msg_hidden, turn=turn)
        self.sample_msg(params, outcomes)
        return outcomes

    def delay(self, sources=None):
        pass

    def sample_con(self, params, turn=0, sources=None):
        raise NotImplementedError()

    def sample_msg(self, params, outcomes):
        raise NotImplementedError()

    @staticmethod
    def _sample_bernoulli(params):
        dist = Bernoulli(logits=params)
        return proper_squeeze(dist.sample((1, )))


class SimulatedSeller(SimulatedActor):
    def __init__(self, model=None, rej_price=0, accept_price=0):
        super(SimulatedSeller, self).__init__(model=model)
        self.rej_price = rej_price
        self.accept_price = accept_price

    def sample_con(self, params, turn=0, sources=None):
        con = categorical_sample(params, 1) / 100
        if con == 0:
            rej = 1
        else:
            rej = 0
        split = get_split(con)
        auto, exp = 0, 0
        norm = 1 - con * sources[O_OUTCOMES_MAP][NORM_POS] - \
                   (1 - sources[L_OUTCOMES_MAP][NORM_POS]) * (1 - con)
        return torch.tensor([con, norm, split, 0, rej, auto, exp]).float()

    def sample_msg(self, params, outcomes):
        if outcomes[4] != 1:
            outcomes[3] = SimulatedActor._sample_bernoulli(params)


class SimulatedBuyer(SimulatedActor):
    def __init__(self, model=None):
        super(SimulatedBuyer, self).__init__(model=model)

    def sample_con(self, params, turn=0, sources=None):
        # resample until non reject on turn 1
        if turn != 7:
            dist = Categorical(logits=params)
            if turn == 1:
                sample = torch.zeros(1)
                while sample == 0:
                    sample = dist.sample((1, ))
                con = proper_squeeze(sample.float() / 100)
            else:
                con = proper_squeeze(dist.sample((1, )).float())
        else:
            con = SimulatedActor._sample_bernoulli(params)
        split = get_split(con)
        norm = (1 - sources[O_OUTCOMES_MAP][2]) * con + sources[L_OUTCOMES_MAP][2] * (1 - con)
        return torch.tensor([con, norm, split, 0]).float()

    def sample_msg(self, params, outcomes):
        outcomes[3] = SimulatedActor._sample_bernoulli(params)