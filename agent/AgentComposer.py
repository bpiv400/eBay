import math
import torch
import numpy as np
from featnames import INT_REMAINING
from constants import TURN_FEATS, BYR, SLR
from featnames import (OUTCOME_FEATS, MONTHS_SINCE_LSTG, BYR_HIST)
from utils import load_sizes, load_featnames
from rlenv.const import *
from agent.const import FEAT_TYPE, CON_TYPE, ALL_FEATS
from agent.util import get_con_set, get_agent_name
from rlenv.Composer import Composer


class AgentComposer(Composer):
    def __init__(self, cols=None, agent_params=None):
        super().__init__(cols)
        # parameters
        self.agent_params = agent_params

        self.sizes['agent'] = self._build_agent_sizes()
        self.x_lstg_cols = list(cols)
        self.turn_inds = None

        # verification
        self.verify_agent()

    @property
    def byr(self):
        return self.agent_params['role'] == BYR

    @property
    def delay(self):
        return self.agent_params[DELAY]

    @property
    def hist(self):
        if self.byr:
            return self.agent_params[BYR_HIST]
        else:
            raise NotImplementedError('No hist attribute for seller')

    @property
    def groupings(self):
        return list(self.sizes['agent']['x'].keys())

    @property
    def feat_type(self):
        return self.agent_params[FEAT_TYPE]

    @property
    def con_type(self):
        return self.agent_params[CON_TYPE]

    def _build_agent_sizes(self):
        sizes = load_sizes(get_agent_name(byr=self.byr, delay=self.delay,
                                          policy=True))
        sizes['out'] = len(get_con_set(self.con_type, byr=self.byr,
                                       delay=self.delay))
        return sizes

    def _update_turn_inds(self, turn):
        if not self.byr:
            inds = np.zeros(2, dtype=np.float32)
        else:
            inds = np.zeros(3, dtype=np.float32)
        active = math.floor((turn - 1) / 2)
        if active < inds.shape[0]:
            inds[active] = 1
        self.turn_inds = inds

    def build_input_dict(self, model_name=None, sources=None, turn=None):
        # build agent input if model name isn't given
        if model_name is None:
            return self._build_agent_dict(sources=sources, turn=turn)
        else:
            return super().build_input_dict(model_name=model_name, sources=sources,
                                            turn=turn)

    def _build_agent_dict(self, sources=None, turn=None):
        obs_dict = dict()
        self._update_turn_inds(turn)
        for set_name in self.agent_sizes['x'].keys():
            if set_name == LSTG_MAP:
                obs_dict[set_name] = self._build_agent_lstg_vector(sources=sources)
            elif set_name[:-1] == 'offer':
                obs_dict[set_name] = self._build_agent_offer_vector(offer_vector=sources[set_name])
            else:
                obs_dict[set_name] = torch.from_numpy(sources[set_name]).float()
        return obs_dict

    def _build_agent_lstg_vector(self, sources=None):
        feats = []
        solo_feats = [sources[MONTHS_SINCE_LSTG], sources[BYR_HIST]]

        # set base, add clock features if buyer and thread count if slr
        if self.byr:
            feats.append(sources[LSTG_MAP][:-4])
            feats.append(sources[CLOCK_MAP][:-2])
        else:
            feats.append(sources[LSTG_MAP])
            solo_feats.append(sources[OFFER_MAPS[1]][THREAD_COUNT_IND] + 1)

        # append solo feats and turn indicators
        solo_feats = np.array(solo_feats)
        feats.append(solo_feats)
        feats.append(self.turn_inds)

        # append remaining for delay models
        if self.delay:
            feats.append(np.array([sources[INT_REMAINING]]))

        # concatenate all features into lstg vector and convert to tensor
        lstg = np.concatenate(feats)
        lstg = lstg.astype(np.float32)
        lstg = torch.from_numpy(lstg).squeeze().float()
        return lstg

    def _build_agent_offer_vector(self, offer_vector=None):
        if not self.byr and self.feat_type == ALL_FEATS:
            full_vector = offer_vector
        else:
            full_vector = np.concatenate([offer_vector[:TIME_START_IND],
                                          offer_vector[TIME_END_IND:]])
        full_vector = torch.from_numpy(full_vector).squeeze().float()
        return full_vector

    def verify_agent(self):
        # exclude auto accept / reject features from buyer shared lstgs
        lstg_sets = self.lstg_sets.copy()
        if self.byr:
            lstg_sets[LSTG_MAP] = lstg_sets[LSTG_MAP][:-4]
        # verify shared lstg features
        model = get_agent_name(delay=self.delay, byr=self.byr, policy=True)
        Composer.verify_lstg_sets_shared(model, self.x_lstg_cols, lstg_sets)
        # verify appended features
        agent_feats = load_featnames(model)
        lstg_append = Composer.remove_shared_feats(agent_feats[LSTG_MAP], lstg_sets[LSTG_MAP])
        self.verify_lstg_append(lstg_append=lstg_append)
        # verify offer feats
        offer_feats = agent_feats['offer']
        self.verify_agent_offer(offer_feats=offer_feats)

    def verify_lstg_append(self, lstg_append=None):
        start_index = 0
        # ensure the first appended buyer features are day-wise clock feats
        if self.byr:
            AgentComposer.verify_sequence(lstg_append, CLOCK_FEATS[:-2], 0)
            start_index += len(CLOCK_FEATS)

        # in all cases check that months_since_lstg and byr_hist are next
        assert lstg_append[start_index] == MONTHS_SINCE_LSTG
        assert lstg_append[start_index + 1] == BYR_HIST

        # for the seller, check that next feature is thread count
        if not self.byr:
            assert lstg_append[start_index + 2] == THREAD_COUNT
            start_index += 3
            agent_role = SLR
        else:
            start_index += 2
            agent_role = BYR

        # ensure turn indicators are next in all cases
        turn_feats = TURN_FEATS[agent_role]
        AgentComposer.verify_sequence(lstg_append, turn_feats, start_index)
        start_index += len(turn_feats)

        # for the delay agents, check that the last feature is int remaining
        if self.delay:
            assert lstg_append[start_index] == INT_REMAINING
            start_index += 1

        # check that all appended features have been exhausted
        assert len(lstg_append) == start_index

    def verify_agent_offer(self, offer_feats=None):
        if not self.byr and self.feat_type == ALL_FEATS:
            assumed_feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS
        else:
            assumed_feats = CLOCK_FEATS + OUTCOME_FEATS
        AgentComposer.verify_all_feats(assumed_feats=assumed_feats, model_feats=offer_feats)
        last_turn = 6 if self.byr or not self.delay else 5
        for turn in range(1, 8):
            if turn <= last_turn:
                assert 'offer{}'.format(turn) in self.agent_sizes['x']
            else:
                assert 'offer{}'.format(turn) not in self.agent_sizes['x']

    @property
    def agent_sizes(self):
        return self.sizes['agent']
