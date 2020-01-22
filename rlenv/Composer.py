import torch
import numpy as np, pandas as pd
from rlenv.env_consts import *
from rlenv.env_utils import load_featnames, model_str, load_sizes
from featnames import (OUTCOME_FEATS, CLOCK_FEATS, TIME_FEATS,
                       BYR_TURN_INDS, SLR_TURN_INDS, MONTHS_SINCE_LSTG,
                       BYR_HIST, INT_REMAINING, MONTHS_SINCE_LAST,
                       THREAD_COUNT)
from constants import ARRIVAL_PREFIX


class Composer:
    """
    Class for composing inputs to interface from various input streams
    """
    def __init__(self, cols):
        self.lstg_sets = Composer.build_lstg_sets(cols)
        self.intervals = self.make_intervals()
        self.offer_feats = Composer.build_offer_feats()
        self.sizes = Composer.make_sizes()
        # TODO

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
        featnames = load_featnames(ARRIVAL_MODEL)
        featnames[LSTG_MAP] = [feat for feat in featnames[LSTG_MAP] if feat in x_lstg_cols]
        for model in MODELS:
            # verify all x_lstg based sets contain the same features in the same order
            Composer.verify_lstg_sets_shared(model, x_lstg_cols, featnames)
            if model in OFFER_MODELS and DELAY not in model:
                Composer.verify_offer_append(model, featnames[LSTG_MAP])
            elif model in OFFER_MODELS:
                Composer.verify_delay_append(model, featnames[LSTG_MAP])
            elif model == ARRIVAL_MODEL:
                Composer.verify_arrival_append(featnames[LSTG_MAP])
            else:
                Composer.verify_hist_append(featnames[LSTG_MAP])
        return featnames

    @staticmethod
    def build_offer_feats():
        shared_feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS
        for model in OFFER_MODELS:
            if SLR_PREFIX in model:
                turn_feats = SLR_TURN_INDS
            else:
                turn_feats = BYR_TURN_INDS
            full_feats = shared_feats + turn_feats
            model_feats = load_featnames(model)['offer']
            assert len(full_feats) == len(model_feats)
            for exp_feat, model_feat in zip(full_feats, model_feats):
                assert exp_feat == model_feat
        return shared_feats

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
            input_dict[grouping_name] = x_lstg.loc[feats].values
        return input_dict

    @staticmethod
    def _build_input_vector(maps, size, sources):
        """
        Helper method that composes a model's input vector given a dictionaries of
        the relevant input maps and  sources

        :param maps: dictionary containing input maps
        :param sources: dictionary containing tensors from the environment that contain
        features the model expects in the input
        :return t: (batch_size x maps[SIZE]) tensor to be passed to a simulator model
        """
        t = torch.zeros(1, size).float()
        # other features
        for map_name, curr_map in maps.items():
            try:
                t[0, curr_map] = torch.from_numpy(sources[map_name][curr_map.index].values).float()
            except RuntimeError as e:
                Composer.catch_input_error(e, t, curr_map, sources, map_name)
                raise RuntimeError()
        return x

    def build_input_dict(self, model_name, sources=None):
        """
        Public method that composes input vectors (x_time and x_fixed) from tensors in the
        environment

        :param name: str giving the name of the focal model
        :param sources: dictionary containing tensors from the environment that contain
        features the model expects in the input
        :return: 2-tuple of x_fixed, x_time. If not recurrent, x_time = None. If fixed=False,
        x_fixed = None
        """
        input_dict = dict()
        fixed_maps = self.maps[model_name]  # dict
        fixed_sizes = self.sizes[model_name]['x']  # dict
        for input_set in fixed_maps.keys():
            input_dict[input_set] = Composer._build_input_vector(
                fixed_maps[input_set], fixed_sizes[input_set], sources)
        return input_dict

    def make_intervals(self):
        ints = {
            BYR_PREFIX: self.sizes[model_str(DELAY, byr=True)][INTERVAL],
            '{}_{}'.format(BYR_PREFIX, 7): self.sizes[model_str(DELAY, byr=True)][INTERVAL],
            SLR_PREFIX: self.sizes[model_str(DELAY, byr=False)][INTERVAL],
            ARRIVAL_PREFIX: self.sizes[ARRIVAL_MODEL][INTERVAL]
        }
        return ints

    @property
    def feat_counts(self):
        counts = dict()
        for set_name, feats in self.feat_sets.items():
            counts[set_name] = len(feats)
        return counts

    @property
    def x_lstg(self):
        return self.feat_sets[LSTG_MAP]

    @staticmethod
    def catch_input_error(e, t, curr_map, sources, map_name):
        print('NAME')
        print(e)
        print(map_name)
        print('stored map: {}'.format(curr_map.dtype))
        print('stored map size: {}'.format(curr_map.dtype))
        print('sourced map: {}'.format(sources[map_name].dtype))
        print('t: {}'.format(t.dtype))

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
        if SLR_PREFIX in model:
            turn_inds = SLR_TURN_INDS
        else:
            turn_inds = BYR_TURN_INDS
        assert len(model_feats[2:]) == len(turn_inds)
        for model_feat, turn_feat in zip(model_feats, turn_inds):
            assert model_feat == turn_feat

    @staticmethod
    def verify_delay_append(model, shared_feats):
        model_feats = load_featnames(model)[LSTG_MAP]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        assert model_feats[0] == MONTHS_SINCE_LSTG
        assert model_feats[1] == BYR_HIST
        if SLR_PREFIX in model:
            turn_inds = SLR_TURN_INDS
        else:
            turn_inds = BYR_TURN_INDS
        Composer.verify_sequence(model_feats, turn_inds, 2)
        assert model_feats[-1] == INT_REMAINING

    @staticmethod
    def verify_arrival_append(shared_feats):
        model_feats = load_featnames(ARRIVAL_MODEL)[LSTG_MAP]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        Composer.verify_sequence(model_feats, CLOCK_FEATS, 0)
        next_ind = len(CLOCK_FEATS)
        assert model_feats[next_ind] == MONTHS_SINCE_LSTG
        assert model_feats[next_ind + 1] == MONTHS_SINCE_LAST
        assert model_feats[next_ind + 2] == THREAD_COUNT

    @staticmethod
    def verify_hist_append(shared_feats):
        model_feats = load_featnames(BYR_HIST_MODEL)[LSTG_MAP]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        assert model_feats[0] == MONTHS_SINCE_LSTG
        Composer.verify_sequence(model_feats, CLOCK_FEATS, 1)
        Composer.verify_sequence(model_feats, TIME_FEATS, 1 + len(CLOCK_FEATS))


class AgentComposer(Composer):
    def __init__(self, params, rebuild=False):
        pass
