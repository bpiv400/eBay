import torch
import numpy as np, pandas as pd
from rlenv.env_consts import *
from rlenv.env_utils import load_featnames, load_sizes, model_str
from featnames import TURN_FEATS
from constants import ARRIVAL_PREFIX


class Composer:
    """
    Class for composing inputs to interface from various input streams
    """
    def __init__(self, cols):
        self.maps, self.sizes, self.feat_sets = \
            Composer.build_models(cols)

    @staticmethod
    def build_models(cols):
        """
        creates a dictionary mapping
        """
        maps = dict()
        sizes = dict()
        x_lstg_cols = list(cols)
        thread_cols = Composer._get_cols(x_lstg_cols)
        feat_sets = {
            THREAD_MAP: thread_cols,
            LSTG_MAP: x_lstg_cols,
            TURN_IND_MAP: TURN_FEATS,
        }
        Composer._check_feat_sets(feat_sets)
        for model in MODELS:
            maps[model], sizes[model] = Composer._build_model_maps(model, feat_sets)
        return maps, sizes, feat_sets

    @staticmethod
    def _get_cols(x_lstg_cols):
        """
        Creates lists of thread features and turn indicator features
        :param x_lstg_cols: list of x_lstg cols
        :return:
        """
        thread_cols = set()
        for mod in MODELS:
            curr_feats = load_featnames(mod)
            for feat_type, feat_set in curr_feats['x'].items():
                [thread_cols.add(feat) for feat in feat_set]
        # exclude turn indicators and consts from thread cols
        for feat_set in [x_lstg_cols, TURN_FEATS]:
            for feat in feat_set:
                if feat in thread_cols:
                    thread_cols.remove(feat)
        return list(thread_cols)

    @staticmethod
    def check_exclusive(x, y, model_name):
        if len(x.intersection(y)) > 0:
            if model_name is not None:
                print('model: {}'.format(model_name))
            print('intersection: {}'.format(x.intersection(y)))
            raise RuntimeError('time cols and thread cols not mutually exclusive')

    @staticmethod
    def _build_model_maps(model, feat_sets):
        maps = dict()
        featnames = load_featnames(model)
        sizes = load_sizes(model)
        for set_name, input_set in featnames['x'].items():
            maps[set_name] = Composer._build_set_maps(input_set, feat_sets, size=sizes['x'][set_name])
        clipped_sizes = sizes.copy()
        del clipped_sizes['out']
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
        :return: batch_size x maps[SIZE] tensor to be passed to a simulator model
        """
        x = torch.zeros(1, size).float()
        # other features
        for map_name, curr_map in maps.items():
            try:
                x[0, curr_map] = torch.from_numpy(sources[map_name][curr_map.index].values).float()
            except RuntimeError as e:
                Composer.catch_input_error(e, x, curr_map, sources, map_name)
                raise RuntimeError()
        return x

    def build_input_dict(self, model_name, sources=None):
        """
        Public method that composes input vectors (x_time and x_fixed) from tensors in the
        environment

        :param model_name: str giving the name of the focal model
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

    @property
    def interval_attrs(self):
        intervals = {
            BYR_PREFIX: self.sizes[model_str(DELAY, byr=True)][INTERVAL],
            '{}_{}'.format(BYR_PREFIX, 7): self.sizes[model_str(DELAY, byr=True)][INTERVAL],
            SLR_PREFIX: self.sizes[model_str(DELAY, byr=False)][INTERVAL],
            ARRIVAL_PREFIX: self.sizes[NUM_OFFERS_MODEL][INTERVAL]
        }
        t7_int_count = self.sizes[model_str(DELAY, byr=True)]['{}_{}'.format(INTERVAL_COUNT, 7)]
        interval_counts = {
            BYR_PREFIX: self.sizes[model_str(DELAY, byr=True)][INTERVAL_COUNT],
            SLR_PREFIX: self.sizes[model_str(DELAY, byr=False)][INTERVAL_COUNT],
            ARRIVAL_PREFIX: self.sizes[NUM_OFFERS_MODEL][INTERVAL_COUNT],
            '{}_{}'.format(BYR_PREFIX, 7): t7_int_count
        }
        return {
            INTERVAL: intervals,
            INTERVAL_COUNT: interval_counts
        }

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
    def catch_input_error(e, x, curr_map, sources, map_name):
        print('NAME')
        print(e)
        print(map_name)
        print('stored map: {}'.format(curr_map.dtype))
        print('stored map size: {}'.format(curr_map.dtype))
        print('sourced map: {}'.format(sources[map_name].dtype))
        print('x: {}'.format(x.dtype))


class AgentComposer(Composer):
    def __init__(self, params, rebuild=False):
        pass
