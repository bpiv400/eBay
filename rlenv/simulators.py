from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch
from rlenv.env_utils import proper_squeeze, categorical_sample, get_split
from rlenv.composer.maps import O_OUTCOMES_MAP, L_OUTCOMES_MAP
from rlenv.env_consts import (NORM_POS, CON_POS, MSG_POS, REJ_POS, DAYS_POS, DELAY_POS,
                              SLR_OUTCOMES, SPLIT_POS, BYR_OUTCOMES)


class SimulatedActor:
    def __init__(self, model=None):
        self.delay_hidden = None
        self.con_hidden = None
        self.msg_hidden = None
        self.model = model

    def make_offer(self, sources=None, turn=1):
        params, self.con_hidden = self.model.con(sources=sources, hidden=self.con_hidden,
                                                 turn=turn)
        outcomes, sample_msg = self.sample_con(params, sources=sources, turn=turn)
        if turn != 7:
            params, self.msg_hidden = self.model.msg(sources=sources, hidden=self.msg_hidden,
                                                     turn=turn)
        else:
            params = None
        # don't draw msg if there's an acceptance or rejection
        if not sample_msg or turn == 7:
            return outcomes
        else:
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
        outcomes = torch.zeros(len(SLR_OUTCOMES)).float()
        con = categorical_sample(params, 1) / 100
        sample_msg = (con != 0 and con != 1)
        if con == 0:
            outcomes[REJ_POS] = 1
            outcomes[NORM_POS] = sources[L_OUTCOMES_MAP][NORM_POS]
        else:
            outcomes[CON_POS] = con
            outcomes[SPLIT_POS] = get_split(con)
            outcomes[NORM_POS] = 1 - con * sources[O_OUTCOMES_MAP][NORM_POS] - \
                       (1 - sources[L_OUTCOMES_MAP][NORM_POS]) * (1 - con)
        return outcomes, sample_msg

    def sample_msg(self, params, outcomes):
        if outcomes[REJ_POS] != 1:
            outcomes[MSG_POS] = SimulatedActor._sample_bernoulli(params)

    def auto_rej(self, sources, turn):
        _, self.con_hidden = self.model.con(sources=sources, hidden=self.con_hidden,
                                            turn=turn)
        _, self.msg_hidden = self.model.msg(sources=sources, hidden=self.msg_hidden,
                                            turn=turn)
        outcomes = sources[L_OUTCOMES_MAP].clone()
        return outcomes


class SimulatedBuyer(SimulatedActor):
    def __init__(self, model=None):
        super(SimulatedBuyer, self).__init__(model=model)

    def sample_con(self, params, turn=0, sources=None):
        # resample until non reject on turn 1
        outcomes = torch.zeros(len(BYR_OUTCOMES)).float()
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
        sample_msg = (con != 0 and con != 1)
        if sample_msg:
            outcomes[SPLIT_POS] = get_split(con)
            outcomes[CON_POS] = con
        outcomes[NORM_POS] = (1 - sources[O_OUTCOMES_MAP][NORM_POS]) * con + \
                             sources[L_OUTCOMES_MAP][NORM_POS] * (1 - con)
        return outcomes, sample_msg

    def sample_msg(self, params, outcomes):
        outcomes[MSG_POS] = SimulatedActor._sample_bernoulli(params)
