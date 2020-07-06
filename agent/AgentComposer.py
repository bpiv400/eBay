import argparse
import math
import pprint
import torch
import numpy as np
from constants import TURN_FEATS, BYR, SLR, PARTS_DIR, TRAIN_RL
from featnames import OUTCOME_FEATS, MONTHS_SINCE_LSTG, BYR_HIST
from utils import load_sizes, load_featnames
from rlenv.const import *
from rlenv.util import load_chunk
from agent.const import FEAT_TYPE, CON_TYPE, ALL_FEATS, AGENT_PARAMS
from agent.util import get_con_set, get_agent_name, compose_args
from rlenv.Composer import Composer


class AgentComposer(Composer):
    def __init__(self, agent_params=None):
        # create columns
        chunk_path = PARTS_DIR + '{}/chunks/1.gz'.format(TRAIN_RL)
        cols = load_chunk(input_path=chunk_path)[0].columns

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
        return BYR in self.agent_params['name']

    @property
    def delay(self):
        return DELAY in self.agent_params['name']

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
        agent_name = get_agent_name(byr=self.byr, delay=self.delay)
        sizes = load_sizes(agent_name)
        sizes['out'] = len(get_con_set(self.con_type,
                                       byr=self.byr,
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
            feats.append(sources[LSTG_MAP][:4])
            feats.append(sources[LSTG_MAP][6:-4])
            feats.append(sources[CLOCK_MAP][:-2])
        else:
            feats.append(sources[LSTG_MAP])
            solo_feats.append(sources[OFFER_MAPS[1]][THREAD_COUNT_IND] + 1)

        # append solo feats and turn indicators
        solo_feats = np.array(solo_feats)
        feats.append(solo_feats)
        feats.append(self.turn_inds)

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
        lstg_sets = self.lstg_sets.copy()
        if self.byr:
            # verify the lstg_sets[LSTG_MAP][4:6] gives count feats
            assert lstg_sets[LSTG_MAP][4:6] == ['lstg_ct', 'bo_ct']
            # verify lstg_sets[LSTG_MAP][-4:] give auto feats
            auto_feats = ['auto_decline', 'auto_accept',
                          'has_decline', 'has_accept']
            assert lstg_sets[LSTG_MAP][-4:] == auto_feats
            # exclude auto accept / reject features  and lstg_ct + bo_ct
            # from byr shared features
            start_feats = lstg_sets[LSTG_MAP][:4]
            end_feats = lstg_sets[LSTG_MAP][6:-4]
            lstg_sets[LSTG_MAP] = start_feats + end_feats
            # remove slr feats
            del lstg_sets[SLR]
        # verify shared lstg features
        model = get_agent_name(delay=self.delay, byr=self.byr)
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
            thread_clock_feats = ['thread_{}'.format(feat) for feat in CLOCK_FEATS]
            thread_clock_feats = thread_clock_feats[:-2]
            AgentComposer.verify_sequence(lstg_append, thread_clock_feats, 0)
            start_index += len(CLOCK_FEATS[:-2])

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


def main():
    parser = argparse.ArgumentParser()
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    agent_params = vars(parser.parse_args())
    print('Args:')
    pprint.pprint(agent_params)
    composer = AgentComposer(agent_params=agent_params)
    print('Verified inputs correctly.')


if __name__ == '__main__':
    main()
