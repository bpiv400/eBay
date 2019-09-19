import os
import pickle
import pandas as pd
import numpy as np
import torch
from rlenv.env_consts import *
from utils import unpickle
import model_names


class Composer:
    """
    Class for composing inputs to models from various input streams

    """
    def __init__(self, params, rebuild=False):
        composer_path = '{}/{}.pkl'.format(COMPOSER_DIR, params['composer'])
        if not os.path.exists(composer_path) or rebuild:
            self.maps = Composer.build()
            pickle.dump(self.maps, open(composer_path, 'wb'))
        else:
            self.maps = unpickle(composer_path)

    @staticmethod
    def build():
        """
        Creates a dictionary containing tensors that map indices from various data
        sources to the input tensors for all the simulator's models

        The dictionary contains 1 entry for each model (18 entries total).
        Each entry is itself a dictionary.

        Dictionaries corresponding  to feed forward models contain only one entry:
        'fixed'. This entry maps to a dictionary containing all of the
        source -> input vector index maps required to compose the the input vector
        for FF models.

        Dictionaries for recurrent models also contain a 'fixed' entry, which contains a
        dictionary of maps used to compose the input vector to the hidden state initialization
        modules

        Dictionaries for recurrent models additionally contain a 'time' entry, which
        contains a dictionary of maps used to compose the input vector at each timestep

        There are two types of index maps: 1-dimensional "simple" maps and 2-dimensional
        "pair maps". Simple maps only contain the indices in the input vector where each
        element of the source vector belongs. These are used in cases where all elements
        from the source vector are included in the input vector

        Pair maps are used in cases where only some of the source vector's elements are
        included in the input vector. A pair map x maps elements from x[i, 0] in
        the source to x[i, 1] in the input
        :return: dictionary of maps for all models
        """
        output = dict()
        fixed = pd.DataFrame.from_dict(LSTG_COLS, orient='index')
        fixed.rename(columns={0: 'from'}, inplace=True)

        for model_name in model_names.MODELS:
            if model_name in model_names.FEED_FORWARD:
                output[model_name] = Composer._build_ff(model_name, fixed)
            else:
                output[model_name] = Composer._build_recurrent(model_name, fixed)
        return output

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
            if model_names.SLR_PREFIX in model_name:
                other_outcomes = ['{}_other'.format(feat) for feat in BYR_OUTCOMES]
                last_outcomes = ['{}_last'.format(feat) for feat in SLR_OUTCOMES]
                indicators = SLR_TURN_INDS
            else:
                other_outcomes = ['{}_other'.format(feat) for feat in SLR_OUTCOMES]
                last_outcomes = ['{}_last'.format(feat) for feat in BYR_OUTCOMES]
                indicators = BYR_TURN_INDS
            maps[O_OUTCOMES_MAP] = Composer._build_simple_map(other_outcomes, featnames)
            maps[L_OUTCOMES_MAP] = Composer._build_simple_map(last_outcomes, featnames)
            other_clock = ['{}_other'.format(feat) for feat in OFFER_CLOCK_FEATS]
            maps[O_CLOCK_MAP] = Composer._build_simple_map(other_clock, featnames)
            last_clock = ['{}_last'.format(feat) for feat in OFFER_CLOCK_FEATS]
            maps[L_CLOCK_MAP] = Composer._build_simple_map(last_clock, featnames)
            other_time = ['{}_other'.format(feat) for feat in TIME_FEATS]
            maps[O_TIME_MAP] = Composer._build_simple_map(other_time, featnames)
            last_time = ['{}_last'.format(feat) for feat in TIME_FEATS]
            maps[L_TIME_MAP] = Composer._build_simple_map(last_time, featnames)
            maps[TURN_IND_MAP] = Composer._build_simple_map(indicators, featnames)
        if model_name != model_names.DAYS:
            maps[BYR_ATTR_MAP] = Composer._build_simple_map(BYR_ATTRS, featnames)
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
        matches = featnames.loc[featnames.index.isin(targ_feats), 'to']
        matches = matches.reindex(targ_feats, axis='index', copy=True)
        matches = matches.to_numpy().astype(int)
        matches = torch.from_numpy(matches).long()
        assert matches.numel() == len(targ_feats)
        assert matches.unique().numel() == matches.numel()
        assert matches.min() >= 0
        return matches

    @staticmethod
    def _build_time(model_name, featnames):
        """
        Builds time feature maps for recurrent nets

        :param model_name:
        :param featnames: pd.DataFrame containing a column "to" that gives the position
        of each index feature in the input vector
        :return:
        """
        featnames = pd.DataFrame(data={'to': np.arange(len(featnames))}, index=featnames)
        time_maps = dict()
        # simple maps where all elements from the source vector are included in
        # the model's input
        time_maps[CLOCK_MAP] = Composer._build_clock_map(model_name, featnames)
        if model_name != model_names.DAYS:
            time_maps[TIME_MAP] = Composer._build_simple_map(TIME_FEATS, featnames)
        # all models except delay and days
        if model_names.DELAY in model_name:
            time_maps[PERIODS_MAP] = torch.tensor([featnames.loc['period', 'to']])
            time_maps[PERIODS_MAP] = time_maps[PERIODS_MAP].long()
        elif model_name != model_names.DAYS:
            if model_names.SLR_PREFIX in model_name:
                outcomes = SLR_OUTCOMES
                other_outcomes = ['{}_other'.format(feat) for feat in BYR_OUTCOMES]
                last_outcomes = ['{}_last'.format(feat) for feat in SLR_OUTCOMES]
                indicators = SLR_TURN_INDS
            else:
                outcomes = BYR_OUTCOMES
                other_outcomes = ['{}_other'.format(feat) for feat in SLR_OUTCOMES]
                last_outcomes = ['{}_last'.format(feat) for feat in BYR_OUTCOMES]
                indicators = BYR_TURN_INDS

            # other clock
            other_clock = ['{}_other'.format(feat) for feat in OFFER_CLOCK_FEATS]
            time_maps[O_CLOCK_MAP] = Composer._build_simple_map(other_clock, featnames)
            # other time
            other_time = ['{}_other'.format(feat) for feat in TIME_FEATS]
            time_maps[O_TIME_MAP] = Composer._build_simple_map(other_time, featnames)
            # diffs
            diffs = ['{}_diff'.format(feat) for feat in TIME_FEATS]
            time_maps[DIFFS_MAP] = Composer._build_simple_map(diffs, featnames)
            # other diffs
            other_diffs = ['{}_other'.format(feat) for feat in diffs]
            time_maps[O_DIFFS_MAP] = Composer._build_simple_map(other_diffs, featnames)
            # other outcomes
            time_maps[O_OUTCOMES_MAP] = Composer._build_simple_map(other_outcomes, featnames)
            time_maps[TURN_IND_MAP] = Composer._build_simple_map(indicators, featnames)

            # pair maps where only some of the elements from the source vector may be
            # included in the model's input
            time_maps[OUTCOMES_MAP] = Composer._build_pair_map(outcomes, featnames)
            time_maps[L_OUTCOMES_MAP] = Composer._build_pair_map(last_outcomes, featnames)
            # add turn indicator map
        return time_maps

    @staticmethod
    def _build_recurrent(model_name, fixed):
        """
        Creates input indices for recurrent models

        :param model_name:
        :param fixed:
        :return:
        """
        if model_name == model_names.DAYS:
            model_type = model_names.ARRIVAL_PREFIX
            subdir = model_name
        elif model_names.SLR_PREFIX in model_name:
            model_type = model_names.SLR_PREFIX
            subdir = model_name.replace('{}_'.format(model_names.SLR_PREFIX), '')
        else:
            model_type = model_names.BYR_PREFIX
            subdir = model_name.replace('{}_'.format(model_names.BYR_PREFIX), '')

        featnames_path = '{}/{}/{}/{}'.format(MODEL_DIR, model_type,
                                              subdir, FEATNAMES_FILENAME)
        featnames = unpickle(featnames_path)
        fixed_maps = Composer._build_fixed(model_name, fixed, featnames['x_fixed'])
        time_maps = Composer._build_time(model_name, featnames['x_time'])
        sizes_path = '{}/{}/{}/{}'.format(MODEL_DIR, model_type,
                                          subdir, SIZES_FILENAME)
        sizes = unpickle(sizes_path)
        # check both maps for correctness
        maps = {FIXED: fixed_maps, TIME: time_maps}
        maps[FIXED][SIZE] = torch.tensor(sizes[FIXED])
        maps[TIME][SIZE] = torch.tensor(sizes[TIME])
        Composer._check_maps(model_name, maps)
        return maps

    @staticmethod
    def _build_lstg_map(fixed_df, featnames):
        """
        Creates a 2 dimensional numpy array mapping indices in the lstg data file
        to indices in a model's fixed input vector, given the index of the feature
        names for that input vector

        :param fixed_df: dataFrame that maps feature names to their location in
        the lstg data file
        :param featnames: dataFrame that aps feature names to their location
        in a model's input vector
        :return: 2-dimensional np.array
        """
        fixed_map = fixed_df.merge(featnames, how='inner',
                                   left_index=True, right_index=True)
        fixed_map = fixed_map.to_numpy().astype(int)
        fixed_map = torch.from_numpy(fixed_map).long()
        return fixed_map

    @staticmethod
    def _build_clock_map(model_name, featnames):
        if model_name in model_names.FEED_FORWARD:
            clock_feats = FF_CLOCK_FEATS
        elif model_name == model_names.DAYS:
            clock_feats = DAYS_CLOCK_FEATS
        elif model_names.DELAY in model_name:
            clock_feats = DELAY_CLOCK_FEATS
        else:
            clock_feats = OFFER_CLOCK_FEATS
        clock_map = Composer._build_simple_map(clock_feats, featnames)
        return clock_map

    @staticmethod
    def _build_ff(model_name, fixed):
        """
        Creates input indices for feed forward models

        Returns a 2-tuple, where the first element is a 3-tuple
        of np.arrays indicating where each feature
        should be passed in the input vector:
            fixed_map: 2-dimensional np.array where first column contains the
            indices in the lstg dataset where each feature comes from and
            the second column gives where each of those features should be placed
            clock_map: 1-dimensional np.array where each clock feature should
            be placed in the input vector
            indiv_map: np.array giving position of byr_us and byr_hist
        and the second element is an integer giving the number of input features

        :param model_name: str giving name of the model
        :param fixed: pandas dataframe with colnames as index and
        col position as first and only column ('pos')
        :return: tuple containing input maps (see above for description
        for each model)
        """
        featnames_path = '{}/arrival/{}/{}'.format(MODEL_DIR, model_name,
                                                   FEATNAMES_FILENAME)
        size_path = '{}/arrival/{}/{}'.format(MODEL_DIR, model_name,
                                              SIZES_FILENAME)
        sizes = unpickle(size_path)
        input_featnames = unpickle(featnames_path)['x_fixed']
        input_featnames = pd.DataFrame(data={'to': np.arange(len(input_featnames))}, index=input_featnames)
        fixed_map = Composer._build_lstg_map(fixed, input_featnames)
        clock_map = Composer._build_clock_map(model_name, input_featnames)
        assert clock_map.numel() == len(FF_CLOCK_FEATS)
        if model_name == model_names.LOC:
            assert clock_map.numel() + fixed_map.shape[0] == len(input_featnames)
            loc_map = None
            hist_map = None
        else:
            loc_map = input_featnames.loc[input_featnames.index == 'byr_us', 'to']
            loc_map = torch.tensor(loc_map.to_numpy().astype(int)).long()
            if model_name == model_names.HIST:
                assert clock_map.numel() + fixed_map.shape[0] + 1 == len(input_featnames)
                hist_map = None
            elif model_name == model_names.BIN or model_name == model_names.SEC:
                hist_map = input_featnames.loc[input_featnames.index == 'byr_hist', 'to']
                hist_map = torch.tensor(hist_map.to_numpy().astype(int)).long()
                assert clock_map.numel() + fixed_map.shape[0] + 2 == len(input_featnames)
            else:
                raise RuntimeError('Invalid model name. See model_names.py')
        maps = {LSTG_MAP: fixed_map,
                CLOCK_MAP: clock_map,
                BYR_US_MAP: loc_map,
                BYR_HIST_MAP: hist_map,
                SIZE: torch.tensor(sizes[FIXED]).long()}
        maps = {FIXED: maps}
        Composer._check_maps(model_name, maps)
        return maps

    @staticmethod
    def _check_maps(model_name, maps):
        """
        Performs some simple sanity checks on the index maps to ensure
        they're not clearly incorrect
        :param maps: tuple containing input maps generated by
        Composer._build_ff or Composer._build_recurrent
        (recurrent if not)
        :return: True if maps are valid
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

    def days_input(self, consts=None, clock_feats=None, fixed=False):
        """
        Returns input tensors for the days model

        :param consts:
        :param clock_feats:
        :param fixed:
        :return:
        """
        maps = self.maps[model_names.DAYS]
        if fixed:

            fixed_maps = maps[FIXED][LSTG_MAP]
            x_fixed = torch.empty(1, 1, fixed_maps[LSTG_MAP].shape[0])
            x_fixed[:, :, fixed_maps[LSTG_MAP][:, 1]] = consts[fixed_maps[LSTG_MAP][:, 0]]
        else:
            x_fixed = None
        time_maps = maps[TIME]
        x_time = torch.empty(1, 1, time_maps[CLOCK_MAP].shape[0])

        x_time[:, :, time_maps[CLOCK_MAP]] = clock_feats
        return x_fixed, x_time

    def loc_input(self, consts=None, clock_feats=None):
        """
        Returns input tensor for the location model

        :param consts:
        :param clock_feats:
        :return:
        """
        maps_dict = self.maps[model_names.LOC]
        fixed_maps = maps_dict[FIXED]

        x_fixed = torch.empty(1, fixed_maps[SIZE])
        x_fixed[:, fixed_maps[LSTG_MAP][:, 1]] = consts[fixed_maps[LSTG_MAP][:, 0]]
        x_fixed[:, fixed_maps[LSTG_MAP][:, 1]] = clock_feats[fixed_maps[LSTG_MAP][:, 0]]
        return x_fixed

    def hist_input(self, consts=None, clock_feats=None, us=False, foreign=False):
        """
        Returns input tensor for the history model

        :param consts:
        :param clock_feats:
        :param us:
        :param foreign:
        :return: 2-dimensional
        """
        maps = self.maps[model_names.HIST][FIXED]
        if us and foreign:
            x_fixed = torch.empty(2, maps[SIZE]).long()
            x_fixed[:, maps[BYR_US_MAP]] = torch.tensor([0, 1])
        else:
            x_fixed = torch.empty(1, maps[SIZE])
            x_fixed[:, maps[BYR_US_MAP]] = (1 if us else 0)
        x_fixed[:, maps[LSTG_MAP][:, 1]] = consts[maps[LSTG_MAP][:, 1]]
        x_fixed[:, maps[CLOCK_MAP]] = clock_feats
        return x_fixed

    def sec_input(self, consts=None, clock_feats=None, byr_us=None, byr_hist=None):
        """
        Returns the input tensor for the seconds arrival model

        :param consts:
        :param clock_feats:
        :param byr_us:
        :param byr_hist:
        :return:
        """
        return Composer.final_arrival_inputs(self.maps[model_names.SEC],
                                             consts, clock_feats, byr_us,
                                             byr_hist)

    @staticmethod
    def final_arrival_inputs(maps, consts, clock_feats, byr_us, byr_hist):
        """
        Returns the input tensor for the seconds or buy-it-now arrival models

        :param maps:
        :param consts:
        :param clock_feats:
        :param byr_us:
        :param byr_hist:
        :return:
        """
        maps = maps[FIXED]
        x_fixed = torch.empty(len(byr_us), maps[SIZE])
        x_fixed[:, maps[LSTG_MAP][:, 1]] = consts[maps[LSTG_MAP][:, 1]]
        x_fixed[:, maps[CLOCK_MAP]] = clock_feats
        x_fixed[:, maps[BYR_US_MAP]] = byr_us
        x_fixed[:, maps[BYR_HIST_MAP]] = byr_hist
        return x_fixed

    def bin_input(self, consts=None, clock_feats=None, byr_us=None, byr_hist=None):
        """
        Returns the input tensor for the buy it now model

        :param consts:
        :param clock_feats:
        :param byr_us:
        :param byr_hist:
        :return:
        """
        return Composer.final_arrival_inputs(self.maps[model_names.BIN],
                                             consts, clock_feats, byr_us,
                                             byr_hist)







