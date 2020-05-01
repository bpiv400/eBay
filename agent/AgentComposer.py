import math
import torch
import numpy as np
from constants import (TURN_FEATS, BYR_PREFIX, SLR_PREFIX,
                       SLR_INIT, BYR_INIT)
from featnames import (OUTCOME_FEATS, MONTHS_SINCE_LSTG, BYR_HIST)
from utils import load_sizes, load_featnames
from rlenv.env_consts import *
from agent.agent_consts import FEAT_TYPE, CON_TYPE, ALL_FEATS
from agent.agent_utils import get_con_set
from rlenv.Composer import Composer


class AgentComposer(Composer):
    def __init__(self, cols=None, agent_params=None):
        super().__init__(cols)
        # parameters
        self.byr = agent_params['role'] == BYR_PREFIX
        self.delay = agent_params[DELAY]
        self.feat_type = agent_params[FEAT_TYPE]
        self.con_type = agent_params[CON_TYPE]

        self.sizes['agent'] = self._build_agent_sizes()
        self.x_lstg_cols = list(cols)
        self.turn_inds = None

        # verification
        self.verify_agent()

    def _build_agent_sizes(self):
        if not self.byr:
            sizes = load_sizes(SLR_INIT)
        else:
            sizes = load_sizes(BYR_INIT)
        sizes['out'] = len(get_con_set(self.con_type))
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
        if model_name is None:
            return self._build_agent_dict(sources=sources, turn=turn)
        agent_remainder = 1 if self.byr else 0
        agent_models = [CON] if not self.delay else [CON, DELAY]
        if turn is not None and turn % 2 == agent_remainder:
            base_model = model_name[:-1]
            if base_model == MSG:
                return None
            elif base_model in agent_models:
                return self._build_agent_dict(sources=sources, turn=turn)
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
        solo_feats = np.array([sources[MONTHS_SINCE_LSTG], sources[BYR_HIST],
                               sources[OFFER_MAPS[1]][THREAD_COUNT_IND] + 1])
        lstg = np.concatenate([sources[LSTG_MAP], solo_feats, self.turn_inds])
        lstg = lstg.astype(np.float32)
        lstg = torch.from_numpy(lstg).squeeze().float()
        return lstg

    def _build_agent_offer_vector(self, offer_vector=None):
        if not self.byr and self.feat_type == ALL_FEATS:
            full_vector = offer_vector
        else:
            full_vector = np.concatenate([offer_vector[:TIME_START_IND],
                                          offer_vector[TIME_END_IND:]])
        full_vector = np.concatenate([full_vector, self.turn_inds])
        full_vector = torch.from_numpy(full_vector).squeeze().float()
        return full_vector

    def verify_agent(self):
        agent_name = SLR_INIT if not self.byr else BYR_INIT
        Composer.verify_lstg_sets_shared(agent_name, self.x_lstg_cols, self.lstg_sets.copy())
        agent_feats = load_featnames(agent_name)
        lstg_append = Composer.remove_shared_feats(agent_feats[LSTG_MAP], self.lstg_sets[LSTG_MAP])
        AgentComposer.verify_lstg_append(lstg_append=lstg_append, agent_name=agent_name)
        offer_feats = agent_feats['offer']
        AgentComposer.verify_agent_offer(offer_feats=offer_feats, agent_name=agent_name)

    @staticmethod
    def verify_lstg_append(lstg_append=None, agent_name=None):
        # print(lstg_append)
        assert lstg_append[0] == MONTHS_SINCE_LSTG
        assert lstg_append[1] == BYR_HIST
        assert lstg_append[2] == THREAD_COUNT
        turn_feats = TURN_FEATS[agent_name]
        AgentComposer.verify_sequence(lstg_append, turn_feats, 3)
        assert len(lstg_append) == (len(turn_feats) + 3)

    @staticmethod
    def verify_agent_offer(offer_feats=None, agent_name=None):
        if agent_name == SLR_INIT:
            assumed_feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS + TURN_FEATS[agent_name]
        else:
            assumed_feats = CLOCK_FEATS + OUTCOME_FEATS + TURN_FEATS[agent_name]
        AgentComposer.verify_all_feats(assumed_feats=assumed_feats, model_feats=offer_feats)
        last_turn = 6 if SLR_PREFIX in agent_name else 7
        sizes = load_sizes(agent_name)['x']
        for i in range(1, 8):
            if i <= last_turn:
                assert 'offer{}'.format(i) in sizes
            else:
                assert 'offer{}'.format(i) not in sizes

    @property
    def agent_sizes(self):
        return self.sizes['agent']
