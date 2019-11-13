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
from rlenv.env_utils import get_featnames_sizes


class Composer:
    """
    Class for composing inputs to interface from various input streams

    """
    def __init__(self, params, rebuild=False):
        composer_path = '{}{}.pkl'.format(COMPOSER_DIR, params['composer'])
        if not os.path.exists(composer_path) or rebuild:
            self.maps, self.x_lstg_cols = Composer.build()
            pickle.dump(self.maps, open(composer_path, 'wb'))
            pickle.dump(self.x_lstg_cols, open(X_LSTG_COLS_PATH, 'wb'))
        else:
            self.maps = unpickle(composer_path)
            self.x_lstg_cols = unpickle(X_LSTG_COLS_PATH)

    @staticmethod
    def build():
        """
        Creates a dictionary containing tensors that map indices from various data
        sources to the input tensors for all the simulator's interface

        The dictionary contains 1 entry for each model (18 entries total).
        Each entry is itself a dictionary.

        Dictionaries corresponding  to feed forward interface contain only one entry:
        'fixed'. This entry maps to a dictionary containing all of the
        source -> input vector index maps required to compose the the input vector
        for FF interface.

        Dictionaries for recurrent interface also contain a 'fixed' entry, which contains a
        dictionary of maps used to compose the input vector to the hidden state initialization
        modules

        Dictionaries for recurrent interface additionally contain a 'time' entry, which
        contains a dictionary of maps used to compose the input vector at each timestep

        There are two types of index maps: 1-dimensional "simple" maps and 2-dimensional
        "pair maps". Simple maps only contain the indices in the input vector where each
        element of the source vector belongs. These are used in cases where all elements
        from the source vector are included in the input vector

        Pair maps are used in cases where only some of the source vector's elements are
        included in the input vector. A pair map x maps elements from x[i, 0] in
        the source to x[i, 1] in the input
        :return: dictionary of maps for all interface
        """
        output = dict()
        x_lstg_cols = load('{}train_rl/1.gz'.format(constants.REWARDS_DIR))['x_lstg'].columns
        pos = np.arange(len(x_lstg_cols))
        fixed = pd.DataFrame(data={'from': pos}, index=x_lstg_cols)

        for model_name in model_names.MODELS:
            if model_name in model_names.FEED_FORWARD:
                output[model_name] = Composer._build_ff(model_name, fixed)
            else:
                output[model_name] = Composer._build_recurrent(model_name, fixed)
        return output, x_lstg_cols

    @staticmethod
    def _add_fixed_delay_maps(maps, featnames, byr=False):
        """
        Adds maps for feature sets that don't appear in any x_fixed inputs
        other than the delay models

        :param maps: dictionary for x_fixed maps
        :param featnames: dataframe where index give feature names in featnames file
        and 'to' column gives indices of features in input vector
        :param byr: boolean for
        :return:
        """
        if not byr:
            other_outcomes = ['{}_other'.format(feat) for feat in BYR_OUTCOMES]
            last_outcomes = ['{}_last'.format(feat) for feat in SLR_OUTCOMES]
            indicators = SLR_TURN_INDS
        else:
            other_outcomes = ['{}_other'.format(feat) for feat in SLR_OUTCOMES]
            last_outcomes = ['{}_last'.format(feat) for feat in BYR_OUTCOMES]
            indicators = BYR_TURN_INDS
        # past turn outcomes
        maps[O_OUTCOMES_MAP] = Composer._build_simple_map(other_outcomes, featnames)
        maps[L_OUTCOMES_MAP] = Composer._build_simple_map(last_outcomes, featnames)
        # indicators
        maps[TURN_IND_MAP] = Composer._build_pair_map(indicators, featnames)
        # last clock
        last_clock = ['{}_last'.format(feat) for feat in CLOCK_FEATS]
        maps[L_CLOCK_MAP] = Composer._build_simple_map(last_clock, featnames)
        # past turn outcomes
        other_time = ['{}_other'.format(feat) for feat in TIME_FEATS]
        maps[O_TIME_MAP] = Composer._build_simple_map(other_time, featnames)
        last_time = ['{}_last'.format(feat) for feat in TIME_FEATS]
        maps[L_TIME_MAP] = Composer._build_simple_map(last_time, featnames)
        # days since thread
        maps[DAYS_THREAD_MAP] = Composer._build_simple_map(DAYS_SINCE_THREAD, featnames)

    @staticmethod
    def _add_fixed_hist_maps(maps, featnames):
        # months since lstg
        maps[MONTHS_LSTG_MAP] = Composer._build_simple_map(MONTHS_SINCE_LSTG, featnames)
        # time feats raw
        maps[TIME_MAP] = Composer._build_simple_map(TIME_FEATS, featnames)
        # clock
        maps[CLOCK_MAP] = Composer._build_simple_map(CLOCK_FEATS, featnames)

    @staticmethod
    def _build_fixed(model_name, fixed, featnames):
        """
        Build fixed feature maps for recurrent networks

        :param model_name: str giving the name of the focal model (see model_names.py)
        :param fixed: dataframe where features in x_lstg make up the index and the column
        "from" gives the index number of each feature in the lstg tensor used in environment
        :param featnames: pd.DataFrame index containing names of all the features in
        the current model's fixed feature input vector in order
        :return: dictionary containing all necessary maps
        """
        featnames = pd.DataFrame(data={'to': np.arange(len(featnames))},
                                 index=featnames)
        fixed_map = Composer._build_lstg_map(fixed, featnames)
        maps = {
            LSTG_MAP: fixed_map,
            SIZE: torch.tensor(len(featnames)).long()
        }
        if model_names.DELAY in model_name:
            Composer._add_fixed_delay_maps(maps, featnames,
                                           byr=constants.BYR_PREFIX in model_name)
        if model_name == model_names.BYR_HIST:
            Composer._add_fixed_hist_maps(maps, featnames)
        if model_name not in model_names.ARRIVAL:
            maps[BYR_HIST_MAP] = Composer._build_simple_map(BYR_HIST, featnames)
        if model_name != model_names.NUM_OFFERS:
            maps[MONTHS_LSTG_MAP] = Composer._build_simple_map(MONTHS_SINCE_LSTG, featnames)
        return maps

    @staticmethod
    def _build_pair_map(targ_feats, featnames):
        """
        Builds paired feature maps for targ_feats under the assumption
        the features are stored in Event objects in tensors with
        the same order as targ_feats (see build() for description of paired feature map)

        :param targ_feats: list containing names of features in some source vector
        :param featnames: pd.DataFrame containing a column "to" that gives the position
        of each index feature in the input vector
        :return: 2-dimensional tensor where the first column gives the index of each
        feature in the source vector and the second column gives the index of each in
        the input vector
        """
        targ_feats = pd.DataFrame({'from': np.arange(len(targ_feats))}, index=targ_feats)
        targ_feats = targ_feats.merge(featnames, left_index=True, right_index=True)
        targ_feats = targ_feats.to_numpy().astype(int)
        targ_feats = torch.from_numpy(targ_feats).long()
        return targ_feats

    @staticmethod
    def _build_simple_map(targ_feats, featnames):
        """
        Builds feature maps for targ feats under the assumption that
        there's a match in featnames for every feat ure in targ_feats

        :param targ_feats: list containing names of features in some source vector
        :param featnames: pd.DataFrame containing a column "to" that gives the position
        of each index feature in the input vector
        :return: 1-dimensional tensor x containing len(targ_feats) elements, where
        x[i] gives the position of targ_feats[i] in the input vector
        :raises: AssertionError if a feature in targ_feats isn't found in featnames
        """
        if isinstance(targ_feats, str):
            targ_feats = [targ_feats]
        matches = featnames.loc[featnames.index.isin(targ_feats), 'to']
        matches = matches.reindex(targ_feats, axis='index', copy=True)
        matches = matches.to_numpy().astype(int)
        matches = torch.from_numpy(matches).long()
        assert matches.numel() == len(targ_feats)
        assert matches.unique().numel() == matches.numel()
        assert matches.min() >= 0
        return matches

    @staticmethod
    def add_time_offer_maps(maps, featnames, byr=False):
        if not byr:
            outcomes = SLR_OUTCOMES
            other_outcomes = ['{}_other'.format(feat) for feat in BYR_OUTCOMES]
            last_outcomes = ['{}_last'.format(feat) for feat in SLR_OUTCOMES]
            indicators = SLR_TURN_INDS
        else:
            outcomes = BYR_OUTCOMES
            other_outcomes = ['{}_other'.format(feat) for feat in SLR_OUTCOMES]
            last_outcomes = ['{}_last'.format(feat) for feat in BYR_OUTCOMES]
            indicators = BYR_TURN_INDS
        # other clock, other outcomes, other diffs
        other_clock = ['{}_other'.format(feat) for feat in CLOCK_FEATS]
        other_diffs = ['{}_other'.format(feat) for feat in TIME_FEATS]
        maps[O_CLOCK_MAP] = Composer._build_simple_map(other_clock, featnames)
        maps[O_OUTCOMES_MAP] = Composer._build_simple_map(other_outcomes, featnames)
        maps[O_DIFFS_MAP] = Composer._build_simple_map(other_diffs, featnames)
        # turn indicators
        maps[TURN_IND_MAP] = Composer._build_simple_map(indicators, featnames)
        # pair maps for outcomes and last outcomes
        maps[OUTCOMES_MAP] = Composer._build_pair_map(outcomes, featnames)
        maps[L_OUTCOMES_MAP] = Composer._build_pair_map(last_outcomes, featnames)

    @staticmethod
    def _build_time(model_name, featnames):
        """
        Creates input maps for recurrent interface' time step input vectors

        :param model_name: str giving the name of the focal model (see model_names.py)
        "from" gives the index number of each feature in the lstg tensor used in environment
        :param featnames: pd.DataFrame index containing names of all the features in
        the current model's fixed feature input vector in order
        :return: dictionary containing all necessary maps
        """
        featnames = pd.DataFrame(data={'to': np.arange(len(featnames))}, index=featnames)
        time_maps = dict()
        # add clock features, time feats diffs to all x_time
        time_maps[CLOCK_MAP] = Composer._build_simple_map(CLOCK_FEATS, featnames)
        diffs = TIME_FEATS.copy()
        time_maps[DIFFS_MAP] = Composer._build_simple_map(diffs, featnames)
        # add duration to delay and num offers
        if model_names.CON in model_name or model_names.MSG in model_name:
            Composer.add_time_offer_maps(time_maps, featnames, byr=constants.BYR_PREFIX in model_name)
            print(time_maps[OUTCOMES_MAP])
        else:
            time_maps[DUR_MAP] = Composer._build_simple_map(DURATION, featnames)
            # add interval remaining to delay
            if model_names.DELAY in model_name:
                time_maps[INT_REMAIN_MAP] = Composer._build_simple_map(INT_REMAINING, featnames)
        return time_maps

    @staticmethod
    def _build_recurrent(full_name, fixed):
        """
        Creates all input maps for recurrent interface (see build() for details)

        :param full: str giving the name of the focal model (see model_names.py)
        :param fixed: dataFrame that maps feature names to their location in
        the lstg tensor used in the environment
        "from" gives the index number of each feature in the x_lstg tensor used in environment
        :return: dictionary containing two entries, one for all time step input maps
        and one for fixed feature input maps
        """
        print('building {}...'.format(full_name))
        featnames, sizes = get_featnames_sizes(full_name=full_name)
        # build maps
        fixed_maps = Composer._build_fixed(full_name, fixed, featnames['x_fixed'])
        time_maps = Composer._build_time(full_name, featnames['x_time'])
        # check both maps for correctness
        maps = {FIXED: fixed_maps, TIME: time_maps}
        maps[FIXED][SIZE] = torch.tensor(sizes[FIXED])
        maps[TIME][SIZE] = torch.tensor(sizes[TIME])
        Composer._check_maps(full_name, maps)
        return maps

    @staticmethod
    def _build_lstg_map(fixed, featnames):
        """
        Creates a 2 dimensional numpy array mapping indices in the lstg data file
        to indices in a model's fixed input vector, given the index of the feature
        names for that input vector

        :param fixed: dataFrame that maps feature names to their location in
        the lstg feature tensor used in the environment. Contains feature names in
        index and positions in the "from" column
        :param featnames: dataFrame that maps feature names to their location
        in a model's input vector
        :return: 2-dimensional tensor where the first column gives the index of each
        feature in the source vector and the second column gives the index of each in
        the input vector
        """
        fixed_map = fixed.merge(featnames, how='inner',
                                left_index=True, right_index=True)
        # ensure the map is ordered by input and contains all fixed features
        assert len(fixed_map.index) == len(fixed.index)
        from_inds = fixed_map['from'].values
        assert np.all(np.equal(from_inds, np.arange(len(from_inds))))
        fixed_map = fixed_map.to_numpy().astype(int)
        fixed_map = fixed_map[:, 1]
        fixed_map = torch.from_numpy(fixed_map).long()
        return fixed_map

    @staticmethod
    def _build_ff(full_name, fixed):
        """
        Creates input maps for feed forward networks

        :param full_name: str giving the name of the focal model (see model_names.py)
        "from" gives the index number of each feature in the lstg tensor used in environment
        :param fixed: dataFrame that maps feature names to their location in
        the lstg tensor used in the environment
        :return: dictionary containing input maps under the 'fixed' index --
        see build() for details
        """
        print('building {}...'.format(full_name))
        featnames, sizes = get_featnames_sizes(full_name=full_name)
        featnames = featnames['x_fixed']
        maps = Composer._build_fixed(full_name, fixed, featnames)
        maps = {FIXED: maps}
        maps[FIXED][SIZE] = torch.tensor(sizes[FIXED])
        Composer._check_maps(full_name, maps)
        return maps

    @staticmethod
    def _check_maps(model_name, maps):
        """
        Performs sanity checks to ensure that input maps aren't clearly
        incorrect:
        -Each feature maps to a distinct index in the input vectors
        -All indices in the input vector have at least 1 source
        index mapping to them
        -The size of the input maps in sum equals the size of the input
        vector

        Note that the composer could still produce incorrect results
        if the order of the features in the source vectors doesn't
        match the order specified by input lists in env_consts

        :param model_name: str giving the name of the focal model (see model_names.py)
        "from" gives the index number of each feature in the lstg tensor used in environment
        :param maps: dictionary output by Composer._build_ff or Composer._build_recurrent
        :raises AssertionError: when maps are not valid
        """
        print("Checking maps for {}".format(model_name))
        for map_type, curr_maps in maps.items():
            maps_list = []
            for map_name, input_map in curr_maps.items():
                if input_map is None:
                    continue
                assert input_map.dtype == torch.int64
                if map_name != SIZE:
                    if len(input_map.shape) == 2:
                        print('found 1')
                        if ((map_name == L_OUTCOMES_MAP or map_name == OUTCOMES_MAP) and
                           input_map.shape[0] == len(SLR_OUTCOMES)):
                            print("{} model: pair map {} may be simple".format(model_name,
                                                                               map_name))
                        maps_list.append(input_map[:, 1])
                    else:
                        maps_list.append(input_map)
            all_to = torch.cat(maps_list)
            assert len(all_to.shape) == 1
            print(all_to.numel())
            print(all_to.unique().numel())
            assert all_to.unique().numel() == all_to.numel()
            assert all_to.max() == all_to.numel() - 1
            assert all_to.min() == 0
            assert all_to.numel() == curr_maps[SIZE]

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
        x = torch.zeros(batch_size, maps[SIZE])
        # other features
        for map_name, curr_map in maps.items():
            try:
                if map_name == SIZE:
                    continue
                if len(curr_map.shape) == 1:
                    x[:, curr_map] = sources[map_name]
                else:
                    x[:, curr_map[:, 1]] = sources[map_name][curr_map[:, 0]]
                if map_name == OUTCOMES_MAP:
                    print(curr_map)
                    print(curr_map.shape)
            except RuntimeError as e:
                print('NAME')
                print(e)
                print(map_name)
                print('stored map: {}'.format(curr_map))
                print('stored map size: {}'.format(curr_map.shape))
                print('sourced map: {}'.format(sources[map_name]))
                raise RuntimeError()
        return x

    def build_arrival_init(self, x_lstg):
        sources = {
            LSTG_MAP: x_lstg
        }
        x_fixed = Composer._build_input_vector(self.maps[model_names.NUM_OFFERS][FIXED],
                                               sources, 1)
        return x_fixed

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
        else:
            x_fixed = None
        return x_fixed, x_time








