import argparse
import math
import torch
import numpy as np
from constants import BYR_DROP
from featnames import OUTCOME_FEATS, DAYS_SINCE_LSTG, BYR_HIST, \
    TIME_FEATS, CLOCK_FEATS, THREAD_COUNT, TURN_FEATS, SLR, META, LEAF
from rlenv.Composer import Composer
from rlenv.const import LSTG_MAP, OFFER_MAPS, THREAD_COUNT_IND, \
    TIME_START_IND, TIME_END_IND
from utils import load_sizes, load_featnames, get_role


class AgentComposer(Composer):

    def __init__(self, byr=None):
        super().__init__()
        self.byr = byr
        if byr:
            self.byr_idx = [i for i in range(len(self.lstg_sets[LSTG_MAP]))
                            if self.lstg_sets[LSTG_MAP][i] not in BYR_DROP]

        self.agent_name = get_role(byr=self.byr)
        self.sizes['agents'] = load_sizes(self.agent_name)

        # parameters to be set later
        self.turn_inds = None

        # verification
        self.verify_agent()

    def _update_turn_inds(self, turn):
        inds = np.zeros(2, dtype=np.float32)
        active = math.floor((turn - 1) / 2)
        if active < inds.shape[0]:
            inds[active] = 1
        self.turn_inds = inds

    def build_input_dict(self, model_name=None, sources=None, turn=None):
        # build agents input if model name isn't given
        if model_name is None:
            return self._build_agent_dict(sources=sources, turn=turn)
        else:
            return super().build_input_dict(model_name=model_name,
                                            sources=sources,
                                            turn=turn)

    def _build_agent_dict(self, sources=None, turn=None):
        obs_dict = dict()
        self._update_turn_inds(turn)
        for set_name in self.agent_sizes['x'].keys():
            if set_name == LSTG_MAP:
                obs_dict[set_name] = self._build_agent_lstg_vector(sources)
            elif set_name[:-1] == 'offer':
                obs_dict[set_name] = self._build_agent_offer_vector(sources[set_name])
            else:
                obs_dict[set_name] = torch.from_numpy(sources[set_name]).float()
        return obs_dict

    def _build_agent_lstg_vector(self, sources=None):
        feats = [sources[LSTG_MAP]]
        solo_feats = [sources[DAYS_SINCE_LSTG], sources[BYR_HIST]]

        # set base, add thread count if slr
        if self.byr:
            feats = [np.array([feats[0][i] for i in self.byr_idx])]
        else:
            solo_feats.append(sources[OFFER_MAPS[1]][THREAD_COUNT_IND] + 1)

        # append solo feats and turn indicators
        solo_feats = np.array(solo_feats)
        feats.append(solo_feats)
        feats.append(self.turn_inds)

        # concatenate all features into lstg vector and convert to tensor
        lstg = np.concatenate(feats).astype(np.float32)
        lstg = torch.from_numpy(lstg).squeeze().float()
        return lstg

    def _build_agent_offer_vector(self, offer_vector=None):
        if not self.byr:
            full_vector = offer_vector
        else:  # remove time feats
            full_vector = np.concatenate([offer_vector[:TIME_START_IND],
                                          offer_vector[TIME_END_IND:]])
        full_vector = torch.from_numpy(full_vector).squeeze().float()
        return full_vector

    def verify_agent(self):
        lstg_sets = self.lstg_sets.copy()
        if self.byr:
            # exclude auto reject features and lstg_ct + bo_ct
            lstg = [lstg_sets[LSTG_MAP][i] for i in self.byr_idx]
            assert len(set(lstg) & set(BYR_DROP)) == 0
            lstg_sets[LSTG_MAP] = lstg

        # remove slr, meta and leaf feats
        for k in [SLR, META, LEAF]:
            del lstg_sets[k]

        # verify shared lstg features
        self.verify_lstg_sets_shared(self.agent_name, lstg_sets)

        # verify appended features
        agent_feats = load_featnames(self.agent_name)
        lstg_append = Composer.remove_shared_feats(agent_feats[LSTG_MAP],
                                                   lstg_sets[LSTG_MAP])
        self.verify_lstg_append(lstg_append=lstg_append)
        # verify offer feats
        offer_feats = agent_feats['offer']
        self.verify_agent_offer(offer_feats=offer_feats)

    def verify_lstg_append(self, lstg_append=None):
        # in all cases check that months_since_lstg and byr_hist are next
        assert lstg_append[0] == DAYS_SINCE_LSTG
        assert lstg_append[1] == BYR_HIST

        # for the seller, check that next feature is thread count
        if not self.byr:
            assert lstg_append[2] == THREAD_COUNT
            start_index = 3
        else:
            start_index = 2

        # ensure turn indicators are next in all cases
        turn_feats = TURN_FEATS[get_role(self.byr)]
        AgentComposer.verify_sequence(lstg_append, turn_feats, start_index)
        start_index += len(turn_feats)

        # check that all appended features have been exhausted
        assert len(lstg_append) == start_index

    def verify_agent_offer(self, offer_feats=None):
        if not self.byr:
            assumed_feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS
        else:
            assumed_feats = CLOCK_FEATS + OUTCOME_FEATS
        AgentComposer.verify_all_feats(assumed_feats=assumed_feats,
                                       model_feats=offer_feats)
        last_turn = 5 if self.byr else 6
        for turn in range(1, 8):
            if turn <= last_turn:
                assert 'offer{}'.format(turn) in self.agent_sizes['x']
            else:
                assert 'offer{}'.format(turn) not in self.agent_sizes['x']

    @property
    def agent_sizes(self):
        return self.sizes['agents']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', 'store_true')
    byr = parser.parse_args().byr
    AgentComposer(byr=byr)
    print('Verified inputs correctly.')


if __name__ == '__main__':
    main()
