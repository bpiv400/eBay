import math
from collections import OrderedDict
import torch
import numpy as np
import pandas as pd
from constants import (MODELS, OFFER_MODELS, SLR_PREFIX, FIRST_ARRIVAL_MODEL,
                       INTERARRIVAL_MODEL, BYR_HIST_MODEL)
from featnames import (OUTCOME_FEATS,
                       MONTHS_SINCE_LSTG, BYR_HIST,
                       INT_REMAINING, MONTHS_SINCE_LAST)
from utils import load_sizes, load_featnames
from rlenv.env_consts import *
from rlenv.env_utils import model_str
from agent.agent_consts import FEAT_TYPE, CON_TYPE, ALL_FEATS, NO_TIME
from agent.agent_utils import get_con_set


class Composer:
    """
    Class for composing inputs to interface from various input streams
    """
    def __init__(self, cols):
        self.sizes = Composer.make_sizes()
        self.lstg_sets = self.build_lstg_sets(cols)
        self.intervals = self.make_intervals()

    @staticmethod
    def make_sizes():
        output = dict()
        for model in MODELS:
            output[model] = load_sizes(model)
        return output

    @staticmethod
    def build_lstg_sets(x_lstg_cols):
        """
        Constructs a dictionary containing the input groups constructed
        from the features in x_lstg

        :param x_lstg_cols: pd.Index containing names of features in x_lstg
        :return: dict
        """
        x_lstg_cols = list(x_lstg_cols)
        featnames = load_featnames(FIRST_ARRIVAL_MODEL)
        featnames[LSTG_MAP] = [feat for feat in featnames[LSTG_MAP] if feat in x_lstg_cols]
        featnames[SLR_PREFIX] = load_featnames(model_str(CON, turn=2))[SLR_PREFIX]
        for model in MODELS:
            # verify all x_lstg based sets contain the same features in the same order
            Composer.verify_lstg_sets_shared(model, x_lstg_cols, featnames.copy())
            if model in OFFER_MODELS:
                Composer.verify_offer_feats(model)
                if DELAY not in model:
                    Composer.verify_offer_append(model, featnames[LSTG_MAP])
                else:
                    Composer.verify_delay_append(model, featnames[LSTG_MAP])
            elif model == FIRST_ARRIVAL_MODEL:
                Composer.verify_first_arrival_append(featnames[LSTG_MAP])
            elif model == INTERARRIVAL_MODEL:
                Composer.verify_interarrival_append(featnames[LSTG_MAP])
            else:
                Composer.verify_hist_append(featnames[LSTG_MAP])
        return featnames

    @staticmethod
    def verify_offer_feats(model):
        turn = int(model[-1])
        if turn % 2 == 0:
            assumed_feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS
        else:
            assumed_feats = CLOCK_FEATS + OUTCOME_FEATS
        model_feats = load_featnames(model)['offer']
        assert len(model_feats) == len(assumed_feats)
        for exp_feat, model_feat in zip(assumed_feats, model_feats):
            assert exp_feat == model_feat
        model_sizes = load_sizes(model)
        for j in range(1, 8):
            if j < turn or (j == turn and DELAY not in model):
                assert 'offer{}'.format(j) in model_sizes['x']
            else:
                assert 'offer{}'.format(j) not in model_sizes['x']

    @staticmethod
    def verify_lstg_sets_shared(model, x_lstg_cols, featnames):
        """
        Ensures that all the input groupings that contain features from x_lstg
        have a common ordering
        :param str model: model name
        :param [str] x_lstg_cols: list of featnames in x_lstg
        :param dict featnames: dictionary containing all x_lstg groupings
        :return: None
        """
        model_featnames = load_featnames(model)
        missing_idx = list()
        # check that all features in LSTG not in x_lstg are appended to the end of LSTG
        for feat in model_featnames[LSTG_MAP]:
            if feat not in x_lstg_cols:
                missing_idx.append(model_featnames[LSTG_MAP].index(feat))
        if len(missing_idx) != 0:
            missing_idx_min = min(missing_idx)
            assert missing_idx_min == len(featnames[LSTG_MAP])
        # remove those missing features
        model_featnames[LSTG_MAP] = [feat for feat in model_featnames[LSTG_MAP] if feat in x_lstg_cols]
        # iterate over all x_lstg features based and check that have same elements in the same order
        for grouping_name, lstg_feats in featnames.items():
            model_grouping = model_featnames[grouping_name]
            assert len(model_grouping) == len(lstg_feats)
            for model_feat, lstg_feat in zip(model_grouping, lstg_feats):
                assert model_feat == lstg_feat
                assert model_feat in x_lstg_cols

    def decompose_x_lstg(self, x_lstg):
        """
        Breaks x_lstg series into separate numpy vectors based on self.lstg_sets that
        serves as basis of
        :param pd.Series x_lstg: fixed feature values
        :return: dict
        """
        input_dict = dict()
        for grouping_name, feats in self.lstg_sets.items():
            input_dict[grouping_name] = x_lstg.loc[feats].values.astype(np.float32)
        return input_dict

    def build_input_dict(self, model_name, sources=None, turn=None):
        """
        Public method that composes input vectors (x_time and x_fixed) from tensors in the
        environment

        :param model_name: str giving the name of the focal model
        :param sources: dictionary containing tensors from the environment that contain
        features the model expects in the input
        :param turn: turn number
        :return: dict
        """
        input_dict = dict()
        fixed_sizes = self.sizes[model_name]['x']  # dict
        for input_set, size in fixed_sizes.items():
            if input_set == LSTG_MAP:
                input_dict[input_set] = self._build_lstg_vector(model_name, sources=sources)
            elif 'offer' == input_set[:-1]:
                input_dict[input_set] = self._build_offer_vector(offer_vector=sources[input_set],
                                                                 byr=turn % 2 != 0)
            else:
                input_dict[input_set] = torch.from_numpy(sources[input_set]).float().unsqueeze(0)
        return input_dict

    def make_intervals(self):
        ints = dict()
        for i in range(1, 8):
            if i == 1:
                ints[i] = self.sizes[FIRST_ARRIVAL_MODEL][INTERVAL]
            else:
                ints[i] = self.sizes[model_str(DELAY, turn=i)][INTERVAL]
        return ints

    @staticmethod
    def _build_offer_vector(offer_vector, byr=False):
        if not byr:
            full_vector = offer_vector
        else:
            full_vector = np.concatenate([offer_vector[:TIME_START_IND],
                                          offer_vector[TIME_END_IND:]])
        return torch.from_numpy(full_vector).unsqueeze(0).float()

    @staticmethod
    def _build_lstg_vector(model_name, sources=None):
        if model_name == INTERARRIVAL_MODEL:
            solo_feats = np.array([sources[MONTHS_SINCE_LSTG], sources[MONTHS_SINCE_LAST],
                                   sources[THREAD_COUNT]])
            lstg = np.concatenate([sources[LSTG_MAP], sources[CLOCK_MAP], solo_feats])
        elif model_name == FIRST_ARRIVAL_MODEL:
            # append nothing
            lstg = sources[LSTG_MAP]
        elif model_name == BYR_HIST_MODEL:
            solo_feats = np.array([sources[MONTHS_SINCE_LSTG],
                                   sources[OFFER_MAPS[1]][THREAD_COUNT_IND]])
            lstg = np.concatenate([sources[LSTG_MAP],
                                   sources[OFFER_MAPS[1]][CLOCK_START_IND:CLOCK_END_IND],
                                   solo_feats])
        else:
            solo_feats = [sources[MONTHS_SINCE_LSTG], 
                          sources[BYR_HIST],
                          sources[OFFER_MAPS[1]][THREAD_COUNT_IND] + 1]
            if DELAY in model_name:
                solo_feats += [sources[INT_REMAINING]]
            lstg = np.concatenate([sources[LSTG_MAP], solo_feats])
        lstg = lstg.astype(np.float32)
        return torch.from_numpy(lstg).float().unsqueeze(0)

    @staticmethod
    def remove_shared_feats(model_feats, shared_feats):
        return model_feats[len(shared_feats):]

    @staticmethod
    def verify_sequence(model_feats, seq_feats, start_idx):
        subset = model_feats[start_idx:(len(seq_feats) + start_idx)]
        for model_feat, seq_feat in zip(subset, seq_feats):
            assert model_feat == seq_feat

    @staticmethod
    def verify_offer_append(model, shared_feats):
        model_feats = load_featnames(model)[LSTG_MAP]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        assert model_feats[0] == MONTHS_SINCE_LSTG
        assert model_feats[1] == BYR_HIST
        assert model_feats[2] == THREAD_COUNT
        assert len(model_feats) == 3

    @staticmethod
    def verify_delay_append(model, shared_feats):
        model_feats = load_featnames(model)[LSTG_MAP]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        assert model_feats[0] == MONTHS_SINCE_LSTG
        assert model_feats[1] == BYR_HIST
        assert model_feats[2] == THREAD_COUNT
        assert model_feats[3] == INT_REMAINING
        assert len(model_feats) == 4

    @staticmethod
    def verify_interarrival_append(shared_feats):
        model_feats = load_featnames(INTERARRIVAL_MODEL)[LSTG_MAP]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        Composer.verify_sequence(model_feats, CLOCK_FEATS, 0)
        next_ind = len(CLOCK_FEATS)
        assert model_feats[next_ind] == MONTHS_SINCE_LSTG
        assert model_feats[next_ind + 1] == MONTHS_SINCE_LAST
        assert model_feats[next_ind + 2] == THREAD_COUNT

    @staticmethod
    def verify_first_arrival_append(shared_feats):
        model_feats = load_featnames(FIRST_ARRIVAL_MODEL)[LSTG_MAP]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        assert len(model_feats) == 0

    @staticmethod
    def verify_hist_append(shared_feats):
        model_feats = load_featnames(BYR_HIST_MODEL)[LSTG_MAP]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        Composer.verify_sequence(model_feats, CLOCK_FEATS, 0)
        assert model_feats[len(CLOCK_FEATS)] == MONTHS_SINCE_LSTG
        assert model_feats[len(CLOCK_FEATS) + 1] == THREAD_COUNT


class AgentComposer(Composer):
    def __init__(self, cols=None, agent_params=None):
        super().__init__(cols)
        # parameters
        self.slr = agent_params[SLR_PREFIX]
        self.delay = agent_params[DELAY]
        self.feat_type = agent_params[FEAT_TYPE]
        self.con_type = agent_params[CON_TYPE]

        self.sizes['agent'] = self._build_agent_sizes()
        self.x_lstg_cols = list(cols)
        self.turn_inds = None

    def _build_agent_sizes(self):
        sizes = OrderedDict()
        num_turns = 6 if self.slr else 7
        for j in range(1, num_turns + 1):
            sizes['offer{}'.format(j)] = self._build_offer_sizes()
        for set_name, feats in self.lstg_sets.items():
            if set_name != SLR_PREFIX or not self.slr:
                if set_name != LSTG_MAP:
                    sizes[set_name] = len(feats)
                else:
                    sizes[set_name] = self._build_lstg_sizes(feats)
        sizes = {
            'x': sizes.copy()
        }
        if not self.delay:
            sizes['out'] = get_con_set(self.con_type).size
        return sizes

    def _update_turn_inds(self, turn):
        if self.slr:
            inds = np.zeros(2, dtype=np.float32)
        else:
            inds = np.zeros(3, dtype=np.float32)
        active = math.floor((turn - 1) / 2)
        if active < inds.shape[0]:
            inds[active] = 1
        self.turn_inds = inds

    def get_obs(self, sources=None, turn=None):
        obs_dict = dict()
        self._update_turn_inds(turn)
        for set_name in self.agent_sizes['x'].keys():
            if set_name == LSTG_MAP:
                # TODO: Update when we know whether to include remaining (i.e. mimic delay or offer)
                obs_dict[set_name] = self._build_agent_lstg_vector(sources=sources)
            elif set_name[:-1] == 'offer':
                obs_dict[set_name] = self._build_agent_offer_vector(offer_vector=sources[set_name])
            else:
                obs_dict[set_name] = torch.from_numpy(sources[set_name]).float()
        return obs_dict

    def _build_agent_lstg_vector(self, sources=None):
        solo_feats = np.array([sources[MONTHS_SINCE_LSTG], sources[BYR_HIST]])
        lstg = np.concatenate([sources[LSTG_MAP], solo_feats, self.turn_inds])
        lstg = lstg.astype(np.float32)
        lstg = torch.from_numpy(lstg).squeeze().float()
        return lstg

    def _build_agent_offer_vector(self, offer_vector=None):
        if self.slr and self.feat_type == ALL_FEATS:
            full_vector = offer_vector
        else:
            full_vector = np.concatenate([offer_vector[:TIME_START_IND],
                                          offer_vector[TIME_END_IND:]])
        full_vector = np.concatenate([full_vector, self.turn_inds])
        full_vector = torch.from_numpy(full_vector).squeeze().float()
        return full_vector

    def _build_offer_sizes(self):
        if not self.slr or self.feat_type == NO_TIME:
            base = len(CLOCK_FEATS) + len(OUTCOME_FEATS)
        else:
            base = len(ALL_OFFER_FEATS)
        # add turn indicators
        if self.slr:
            return base + 2
        else:
            return base + 3

    def _build_lstg_sizes(self, shared_feats):
        # add 2 to shared features for months_since_lstg + byr_hist
        base = len(shared_feats) + 2
        # turn indicates
        base = base + 2 if self.slr else base + 3
        return base

    @property
    def agent_sizes(self):
        return self.sizes['agent']
