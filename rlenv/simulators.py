import pandas as pd
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch
from rlenv.env_utils import proper_squeeze, categorical_sample, get_split, featname
from rlenv.env_consts import ALL_OUTCOMES, REJECT, NORM, CON, SPLIT, MSG, AUTO, EXP
from rlenv.composer.maps import THREAD_MAP


class SimulatedActor:
    def __init__(self, model=None):
        self.delay_hidden = None
        self.model = model

    def make_offer(self, sources=None, turn=None):
        params = self.model.con(sources=sources, turn=turn)
        outcomes, sample_msg = self.sample_con(params, sources=sources, turn=turn)
        # don't draw msg if there's an acceptance or rejection
        if not sample_msg:
            return outcomes
        else:
            params = self.model.msg(sources=sources, turn=turn)
            self.sample_msg(params, outcomes, turn)
        return outcomes

    def init_delay(self, sources):
        self.delay_hidden = self.model.init_delay(sources=sources)

    def delay(self, sources=None):
        params, self.delay_hidden = self.model.delay(sources=sources,
                                                     hidden=self.delay_hidden)
        make_offer = SimulatedActor._sample_bernoulli(params)
        return make_offer

    def sample_con(self, params, turn=0, sources=None):
        raise NotImplementedError()

    def sample_msg(self, params, outcomes, turn):
        raise NotImplementedError()

    @staticmethod
    def _sample_bernoulli(params):
        dist = Bernoulli(logits=params)
        return proper_squeeze(dist.sample((1, ))).numpy()

    @staticmethod
    def prev_norm(sources, turn):
        if turn <= 2:
            prev_norm = 0.0
        else:
            prev_norm = sources[THREAD_MAP][featname(NORM, turn - 2)]
        return prev_norm


class SimulatedSeller(SimulatedActor):
    def __init__(self, model=None):
        super(SimulatedSeller, self).__init__(model=model)

    def sample_con(self, params, turn=None, sources=None):
        outcomes = pd.Series(0.0, index=ALL_OUTCOMES[turn])
        con = (categorical_sample(params, 1) / 100)
        sample_msg = (con != 0 and con != 1)
        # compute previous seller norm or set to 0 if this is the first turn
        prev_slr_norm = SimulatedSeller.prev_norm(sources, turn)
        # handle rejection case
        if con == 0:
            outcomes[featname(REJECT, turn)] = 1
            outcomes[featname(NORM, turn)] = prev_slr_norm
        else:
            outcomes[featname(CON, turn)] = con
            outcomes[featname(SPLIT, turn)] = get_split(con)
            prev_byr_norm = sources[THREAD_MAP][featname(NORM, turn - 1)]
            norm = 1 - con * prev_byr_norm - (1 - prev_slr_norm) * (1 - con)
            outcomes[featname(NORM, turn)] = norm
        return outcomes, sample_msg

    def sample_msg(self, params, outcomes, turn):
        outcomes[featname(MSG, turn)] = SimulatedActor._sample_bernoulli(params)

    @staticmethod
    def rej(sources, turn, expire=False):
        outcomes = pd.Series(0.0, index=ALL_OUTCOMES[turn])
        outcomes[featname(REJECT, turn)] = 1
        outcomes[featname(NORM, turn)] = SimulatedSeller.prev_norm(sources, turn)
        if not expire:
            outcomes[featname(AUTO, turn)] = 1
        else:
            outcomes[featname(EXP, turn)] = 1
        return outcomes


class SimulatedBuyer(SimulatedActor):
    def __init__(self, model=None):
        super(SimulatedBuyer, self).__init__(model=model)

    def sample_con(self, params, turn=None, sources=None):
        # resample until non reject on turn 1
        outcomes = pd.Series(0.0, index=ALL_OUTCOMES[turn])
        if turn != 7:
            dist = Categorical(logits=params)
            if turn == 1:
                sample = torch.zeros(1)
                while sample == 0:
                    # print('sampling concession')
                    sample = dist.sample((1, ))
                con = proper_squeeze(sample.float() / 100).numpy()
                # print('concession: {}'.format(con))
            else:
                con = proper_squeeze(dist.sample((1, )).float()).numpy()
                con = con / 100
        else:
            con = SimulatedActor._sample_bernoulli(params)
        sample_msg = (con != 0 and con != 1)
        if sample_msg:
            outcomes[featname(SPLIT, turn)] = get_split(con)
        outcomes[featname(CON, turn)] = con
        prev_slr_norm = SimulatedBuyer.prev_slr_norm(sources, turn)
        prev_byr_norm = SimulatedBuyer.prev_norm(sources, turn)
        norm = (1 - prev_slr_norm) * con + prev_byr_norm * (1 - con)
        outcomes[featname(NORM, turn)] = norm
        return outcomes, sample_msg

    def sample_msg(self, params, outcomes, turn):
        outcomes[featname(MSG, turn)] = SimulatedActor._sample_bernoulli(params)

    @staticmethod
    def prev_slr_norm(sources, turn):
        if turn == 1:
            return 0.0
        else:
            return sources[THREAD_MAP][featname(NORM, turn - 1)]
