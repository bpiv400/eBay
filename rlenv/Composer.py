import torch
import numpy as np, pandas as pd
from rlenv.env_consts import *
from rlenv.env_utils import load_featnames, load_sizes, model_str
from featnames import OUTCOME_FEATS, CLOCK_FEATS, TIME_FEATS, BYR_TURN_INDS, SLR_TURN_INDS
from constants import ARRIVAL_PREFIX


class Composer:
    """
    Class for composing inputs to interface from various input streams
    """
    def __init__(self, cols):
        self.lstg_sets = Composer.build_lstg_sets(cols)
        self.intervals = self.make_intervals()
        self.offer_feats = Composer.build_offer_feats()

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
            Composer.verify_lstg_sets_shared(model, x_lstg_cols, featnames)
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
    def _build_model_maps(model, feat_sets):
        maps = dict()
        featnames = load_featnames(model)
        sizes = load_sizes(model)
        for set_name, input_set in featnames.items():
            maps[set_name] = Composer._build_set_maps(input_set, feat_sets, size=sizes['x'][set_name])
        clipped_sizes = sizes.copy()
        return maps, clipped_sizes

    @staticmethod
    def _build_set_maps(input_set, feat_sets, size=None):
        output = dict()
        #print('input set: {}'.format(len(input_set)))
        input_set = pd.DataFrame(data={'out':np.arange(len(input_set))},
                                 index=input_set)
        for set_name, feat_list in feat_sets.items():
            if input_set.index.isin(feat_list).any():
                output[set_name] = Composer._build_pair_map(input_set, feat_list)
        Composer._check_set_maps(output, input_set, size)
        return output

    @staticmethod
    def _build_pair_map(input_set, feat_list):
        """
        Builds paired feature maps for targ_feats under the assumption
        the features are stored in Event objects in tensors with
        the same order as targ_feats (see build() for description of paired feature map)

        :param input_set:
        :param feat_list:
        :return:
        """
        input_set = input_set.copy()
        input_set = input_set.loc[input_set.index.isin(feat_list), 'out']
        return input_set

    @staticmethod
    def _check_set_maps(maps, input_set, size):
        """
        Performs sanity checks to ensure that input maps aren't clearly
        incorrect:
        - Each map is a Series
        -Each feature maps to a distinct index in the input vectors
        -All indices in the input vector have at least 1 source
        index mapping to them
        -The size of the input maps in sum equals the size of the input
        vector

        :param input_set:
        :param maps: dictionary output by Composer._build_ff or Composer._build_recurrent
        :raises AssertionError: when maps are not valid
        """
        total = len(input_set)
        indices = list()
        map_feats = list()
        for map_name, input_map in maps.items():
            assert isinstance(input_map, pd.Series)
            indices = indices + list(input_map.values)
            map_feats = map_feats + list(input_map.index)
        assert len(map_feats) == len(indices)
        assert len(indices) == total
        assert min(indices) == 0
        assert max(indices) == (total - 1)
        assert len(indices) == len(set(indices))
        # error checking
        input_feats = set(list(input_set.index))
        # print(len(input_set))
        map_feats = set(map_feats)
        # print('missing from maps: {}'.format(input_feats.difference(map_feats)))
        assert len(indices) == size
        assert input_feats == map_feats

    @staticmethod
    def _check_feat_sets(feat_sets):
        feat_map = dict()
        for _, feat_list in feat_sets.items():
            for feat in feat_list:
                assert feat not in feat_map
                feat_map[feat] = True

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


class AgentComposer(Composer):
    def __init__(self, params, rebuild=False):
        pass
