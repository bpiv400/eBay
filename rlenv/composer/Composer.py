import os
import pickle

from rlenv.composer.maps import *
from compress_pickle import load
import constants
import pandas as pd
import numpy as np
from rlenv.env_consts import *
from utils import unpickle
from rlenv.interface import model_names
from rlenv.env_utils import load_featnames, load_sizes


class Composer:
    """
    Class for composing inputs to interface from various input streams

    """
    def __init__(self, params, rebuild=False):
        composer_path = '{}{}.pkl'.format(COMPOSER_DIR, params['composer'])
        if not os.path.exists(composer_path) or rebuild:
            self.maps, self.feat_sets = Composer.build()
            pickle.dump((self.maps, self.feat_sets), open(composer_path, 'rb'))
        else:
            self.maps, self.feat_sets = unpickle(composer_path)

    @staticmethod
    def build():
        """
        creates a dictionary mapping
        """
        output = dict()
        x_lstg_cols = pickle.load(X_LSTG_COLS_PATH)
        thread_cols, x_time_cols = Composer._get_cols(x_lstg_cols)
        thread_cols, x_time_cols = list(thread_cols), list(x_time_cols)
        feat_sets = {
            THREAD_MAP: thread_cols,
            LSTG_MAP: x_lstg_cols,
            TURN_IND_MAP: TURN_FEATS,
            X_TIME_MAP: x_time_cols,
        }
        Composer._check_feat_sets(feat_sets)
        for model in model_names.MODELS:
            output[model] = Composer._build_model_maps(model, feat_sets)
        return output, feat_sets

    @staticmethod
    def _get_cols(x_lstg_cols):
        thread_cols, x_time_cols = set(), set()
        # iterate over all models accumulating thread features and x_time features
        for mod in model_names.MODELS:
            curr_feats = load_featnames(mod)
            [x_time_cols.add(feat) for feat in curr_feats['x_time']]
            for feat_type, feat_set in curr_feats['x'].items():
                [thread_cols.add(feat) for feat in feat_set]
        # exclude turn indicators
        for feat in TURN_FEATS:
            if feat in thread_cols:
                thread_cols.remove(feat)
            if feat in x_time_cols:
                x_time_cols.remove(feat)
        # exclude features from x_lstg
        for feat in x_lstg_cols:
            if feat in thread_cols:
                thread_cols.remove(feat)
            if feat in x_time_cols:
                x_time_cols.remove(feat)
        # check that x_time and thread cols are mutually exlcusive
        assert len(x_time_cols.intersection(thread_cols)) == 0
        return thread_cols, x_time_cols

    @staticmethod
    def _build_model_maps(model, feat_sets):
        output = dict()
        featnames = load_featnames(model)
        # create input set for x_time
        input_set = featnames['x_time']
        output['x_time'] = Composer._build_set_maps(input_set, feat_sets)
        # ensure only features from x_time contribute to the x_time map
        assert len(output['x_time']) == 1
        output['x'] = dict()
        for set_name, input_set in featnames['x']:
            output['x'][set_name] = Composer._build_set_maps(input_set, feat_sets)
        return output

    @staticmethod
    def _build_set_maps(input_set, feat_sets):
        output = dict()
        input_set = pd.DataFrame(data={'out':np.arange(len(input_set))},
                                 index=input_set)
        for set_name, feat_list in feat_sets.items():
            if input_set.index.isin(feat_list).any():
                output[set_name] = Composer._build_pair_map(input_set, feat_list)
        Composer._check_set_maps(output, input_set)
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
    def _check_set_maps(maps, input_set):
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
        for map_name, input_map in maps.items():
            assert isinstance(input_map, pd.Series)
            indices = indices + list(input_map.values)
        assert len(indices) == total
        assert min(indices) == 0
        assert max(indices) == (total - 1)
        assert len(indices) == len(set(indices))

    @staticmethod
    def _check_feat_sets(feat_sets):
        feat_map = dict()
        for _, feat_list in feat_sets.items():
            for feat in feat_list:
                assert feat not in feat_map
                feat_map[feat] = True

    @staticmethod
    def _build_input_vector(maps, sources, batch_size):
        """
        Helper method that composes a model's input vector given a dictionaries of
        the relevant input maps and  sources

        :param maps: dictionary containing input maps
        :param sources: dictionary containing tensors from the environment that contain
        features the model expects in the input
        :param batch_size: number of examples in the batch
        :return: batch_size x maps[SIZE] tensor to be passed to a simulator model
        """
        x = torch.zeros(batch_size, maps[SIZE]).float()
        # other features
        for map_name, curr_map in maps.items():
            try:
                if map_name == SIZE:
                    continue
                if len(curr_map.shape) == 1:
                    x[:, curr_map] = sources[map_name]
                else:
                    x[:, curr_map[:, 1]] = sources[map_name][curr_map[:, 0]]
            except RuntimeError as e:
                print('NAME')
                print(e)
                print(map_name)
                print('stored map: {}'.format(curr_map.dtype))
                print('stored map size: {}'.format(curr_map.dtype))
                print('sourced map: {}'.format(sources[map_name].dtype))
                print('x: {}'.format(x.dtype))
                raise RuntimeError()
        return x

    def build_arrival_init(self, x_lstg):
        sources = {
            LSTG_MAP: x_lstg
        }
        x_fixed = Composer._build_input_vector(self.maps[model_names.NUM_OFFERS][FIXED],
                                               sources, 1)
        return x_fixed.unsqueeze(0)

    def build_input_vector(self, model_name, sources=None, fixed=False, recurrent=False):
        """
        Public method that composes input vectors (x_time and x_fixed) from tensors in the
        environment

        :param model_name: str giving the name of the focal model (see model_names.py)
        :param sources: dictionary containing tensors from the environment that contain
        features the model expects in the input
        :param fixed: boolean for whether x_fixed needs to be compute
        :param recurrent: boolean for whether the target model is recurrent
        :param size: number of samples in the batch that will be input to the model
        :return: 2-tuple of x_fixed, x_time. If not recurrent, x_time = None. If fixed=False,
        x_fixed = None
        """
        if recurrent:
            x_time = Composer._build_input_vector(self.maps[model_name][TIME], sources, 1)
            x_time = x_time.unsqueeze(0)
        else:
            x_time = None
        if fixed:
            x_fixed = Composer._build_input_vector(self.maps[model_name][FIXED], sources, 1)
            if model_name != model_names.BYR_HIST:
                x_fixed = x_fixed.unsqueeze(0)
        else:
            x_fixed = None
        return x_fixed, x_time
