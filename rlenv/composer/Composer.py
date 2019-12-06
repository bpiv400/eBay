import os
import pickle

from rlenv.composer.maps import *
from compress_pickle import load
import pandas as pd
import numpy as np
from rlenv.env_consts import *
from constants import REWARDS_DIR
from utils import unpickle
from rlenv.interface import model_names
from rlenv.env_utils import load_featnames, load_sizes
from rlenv.composer.fix_featnames import fix_featnames


class Composer:
    """
    Class for composing inputs to interface from various input streams

    """
    def __init__(self, params, rebuild=False):
        composer_path = '{}{}.pkl'.format(COMPOSER_DIR, params['composer'])
        if not os.path.exists(composer_path) or rebuild:
            self.maps, self.sizes, self.feat_sets = Composer.build()
            pickle.dump((self.maps, self.sizes, self.feat_sets), open(composer_path, 'wb'))
        else:
            self.maps, self.sizes, self.feat_sets = unpickle(composer_path)

    @staticmethod
    def build():
        """
        creates a dictionary mapping
        """
        maps = dict()
        sizes = dict()
        x_lstg_path = '{}train_rl/chunks/1.gz'.format(REWARDS_DIR)
        x_lstg_cols = list(load(x_lstg_path)['x_lstg'].columns)
        thread_cols, x_time_cols = Composer._get_cols(x_lstg_cols)
        thread_cols, x_time_cols = list(thread_cols), list(x_time_cols)
        feat_sets = {
            THREAD_MAP: thread_cols,
            LSTG_MAP: x_lstg_cols,
            TURN_IND_MAP: TURN_FEATS,
            X_TIME_MAP: x_time_cols,
        }
        #################################
        # debugging to check turn 7
        feats7 = list()
        for feat in feat_sets[THREAD_MAP]:
            if '_7' in feat:
                feats7.append(feat)
        total7 = ALL_CLOCK_FEATS[7] + ALL_OUTCOMES[7] + ALL_TIME_FEATS[7]
        total7 = set(total7)
        feats7 = set(feats7)
        print('missing in 7: {}'.format(total7.difference(feats7)))
        #################################
        Composer._check_feat_sets(feat_sets)
        for model in model_names.MODELS:
            maps[model], sizes[model] = Composer._build_model_maps(model, feat_sets)
        return maps, sizes, feat_sets

    @staticmethod
    def _get_cols(x_lstg_cols):
        thread_cols, x_time_cols = set(), set()
        # iterate over all models accumulating thread features and x_time features
        for mod in model_names.MODELS:
            curr_feats = load_featnames(mod)
            if 'x_time' in curr_feats:
                [x_time_cols.add(feat) for feat in curr_feats['x_time']]
        prev_int = set()
        for mod in model_names.MODELS:
            curr_feats = load_featnames(mod)

            # TODO: HAVE ETAN USE PROPER FEATNAMES
            curr_feats = fix_featnames(curr_feats)
            ##############################################

            for feat_type, feat_set in curr_feats['x'].items():
                [thread_cols.add(feat) for feat in feat_set]
                # debugging
                if len(x_time_cols.intersection(thread_cols)) > 0:
                    curr_int = x_time_cols.intersection(thread_cols)
                    new_int = curr_int.difference(prev_int)
                    if len(new_int) > 0:
                        print('model: {}'.format(mod))
                        print('feat set: {}'.format(feat_type))
                        print('intersection: {}'.format(new_int))
                        print('')
                        prev_int = prev_int.union(new_int)
                        raise RuntimeError("unexpected intersection between maps")
        # exclude turn indicators
        for feat_set in [x_lstg_cols, TURN_FEATS]:
            for feat in feat_set:
                if feat in thread_cols:
                    thread_cols.remove(feat)
                if feat in x_time_cols:
                    x_time_cols.remove(feat)
        # check that x_time and thread cols are mutually exlcusive
        if len(x_time_cols.intersection(thread_cols)) > 0:
            print('intersection: {}'.format(x_time_cols.intersection(thread_cols)))
        assert len(x_time_cols.intersection(thread_cols)) == 0
        return thread_cols, x_time_cols

    @staticmethod
    def _build_model_maps(model, feat_sets):
        # print(model)
        maps = dict()
        featnames = fix_featnames(load_featnames(model))
        sizes = load_sizes(model)
        # temporary fix
        if 'time' in sizes:
            sizes['x_time'] = sizes['time']
            del sizes['time']
        # create input set for x_time
        if 'x_time' in featnames:
            input_set = featnames['x_time']
            maps['x_time'] = Composer._build_set_maps(input_set, feat_sets,
                                                      size=sizes['x_time'])
            # ensure only features from x_time contribute to the x_time map
            assert len(maps['x_time']) == 1
        maps['x'] = dict()
        for set_name, input_set in featnames['x'].items():
            # print(set_name)
            maps['x'][set_name] = Composer._build_set_maps(input_set, feat_sets,
                                                           size=sizes['x'][set_name])
        clipped_sizes = {
            'x': sizes['x'],
        }
        if 'x_time' in featnames:
            clipped_sizes['x_time'] = sizes['x_time']
        return maps, clipped_sizes

    @staticmethod
    def _build_set_maps(input_set, feat_sets, size=None):
        output = dict()
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
        # print('num maps: {}'.format(len(maps)))
        map_feats = list()
        for map_name, input_map in maps.items():
            assert isinstance(input_map, pd.Series)
            indices = indices + list(input_map.values)
            map_feats = map_feats + list(input_map.index)

        # TODO: DEBUGGING
        # print(total)
        # print(len(indices))
        # input_feats = set(list(input_set.index))
        # map_feats = set(map_feats)
        # print('in input not in maps: {}'.format(input_feats.difference(map_feats)))
        ############################
        assert len(map_feats) == len(indices)
        assert len(indices) == total
        assert min(indices) == 0
        assert max(indices) == (total - 1)
        assert len(indices) == len(set(indices))
        assert len(indices) == size

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
                print('NAME')
                print(e)
                print(map_name)
                print('stored map: {}'.format(curr_map.dtype))
                print('stored map size: {}'.format(curr_map.dtype))
                print('sourced map: {}'.format(sources[map_name].dtype))
                print('x: {}'.format(x.dtype))
                raise RuntimeError()
        return x

    def build_input_vector(self, model_name, sources=None, fixed=False, recurrent=False):
        """
        Public method that composes input vectors (x_time and x_fixed) from tensors in the
        environment

        :param model_name: str giving the name of the focal model (see model_names.py)
        :param sources: dictionary containing tensors from the environment that contain
        features the model expects in the input
        :param fixed: boolean for whether x_fixed needs to be compute
        :param recurrent: boolean for whether the target model is recurrent
        :return: 2-tuple of x_fixed, x_time. If not recurrent, x_time = None. If fixed=False,
        x_fixed = None
        """
        input_dict = dict()
        if recurrent:
            input_dict['x_time'] = Composer._build_input_vector(self.maps[model_name]['x_time'],
                                                                self.sizes[model_name]['x_time'],
                                                                sources)
        if fixed:
            input_dict['x'] = dict()
            fixed_maps = self.maps[model_name]['x']
            fixed_sizes = self.sizes[model_name]['x']
            for input_set in fixed_maps.keys():
                input_dict['x'][input_set] = Composer._build_input_vector(fixed_maps[input_set],
                                                                          fixed_sizes[input_set],
                                                                          sources)
        return input_dict

    @property
    def feat_counts(self):
        counts = dict()
        for set_name, feats in self.feat_sets.items():
            counts[set_name] = len(feats)
        return counts
