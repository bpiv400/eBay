from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch
from rlenv.env_utils import proper_squeeze, categorical_sample, get_split
from rlenv.composer.maps import O_OUTCOMES_MAP, L_OUTCOMES_MAP
from rlenv.env_consts import (NORM_POS, CON_POS, MSG_POS, REJ_POS, EXPIRE_POS,
                              SLR_OUTCOMES, SPLIT_POS, BYR_OUTCOMES, AUTO_POS)


class SimulatedActor:
    def __init__(self, model=None):
        self.delay_hidden = None
        self.model = model

    def make_offer(self, sources=None, turn=1):
        params = self.model.con(sources=sources, turn=turn)
        outcomes, sample_msg = self.sample_con(params, sources=sources, turn=turn)
        # don't draw msg if there's an acceptance or rejection
        if not sample_msg:
            return outcomes
        else:
            params = self.model.msg(sources=sources, turn=turn)
            self.sample_msg(params, outcomes)
        return outcomes

    def delay(self, sources=None):
        params, self.delay_hidden = self.model.delay(sources=sources,
                                                     hidden=self.delay_hidden)
        make_offer = SimulatedActor._sample_bernoulli(params)
        return make_offer

    def init_delay(self, sources):
        self.delay_hidden = self.model.init_delay(sources=sources)

    def sample_con(self, params, turn=0, sources=None):
        raise NotImplementedError()

    def sample_msg(self, params, outcomes):
        raise NotImplementedError()

    @staticmethod
    def _sample_bernoulli(params):
        dist = Bernoulli(logits=params)
        return proper_squeeze(dist.sample((1, )))


class SimulatedSeller(SimulatedActor):
    def __init__(self, model=None):
        super(SimulatedSeller, self).__init__(model=model)

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

    def rej(self, sources, turn, expire=False):
        _, self.con_hidden = self.model.con(sources=sources, hidden=self.con_hidden,
                                            turn=turn)
        _, self.msg_hidden = self.model.msg(sources=sources, hidden=self.msg_hidden,
                                            turn=turn)
        outcomes = torch.zeros(len(SLR_OUTCOMES)).float()
        outcomes[REJ_POS] = 1
        outcomes[NORM_POS] = sources[L_OUTCOMES_MAP][NORM_POS]
        if not expire:
            outcomes[AUTO_POS] = 1
        else:
            outcomes[EXPIRE_POS] = 1
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
                    # print('sampling concession')
                    sample = dist.sample((1, ))
                con = proper_squeeze(sample.float() / 100)
                # print('concession: {}'.format(con))
            else:
                con = proper_squeeze(dist.sample((1, )).float())
                con = con / 100
        else:
            con = SimulatedActor._sample_bernoulli(params)
        sample_msg = (con != 0 and con != 1)
        if sample_msg:
            outcomes[SPLIT_POS] = get_split(con)
        outcomes[CON_POS] = con
        outcomes[NORM_POS] = (1 - sources[O_OUTCOMES_MAP][NORM_POS]) * con + \
                             sources[L_OUTCOMES_MAP][NORM_POS] * (1 - con)
        # print('Norm associated with concession: {}'.format(outcomes[NORM_POS]))
        return outcomes, sample_msg

    def sample_msg(self, params, outcomes):
        outcomes[MSG_POS] = SimulatedActor._sample_bernoulli(params)
