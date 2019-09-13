import os
import pandas as pd
import numpy as np
from rlenv.env_constants import (COMPOSER_FILENAME, LSTG_COLS, INFO_FILENAME, MODEL_DIR,
    ARRIVAL_CLOCK_FEATS)
from utils import unpickle
import model_names

class Composer:
    """
    Class for composing inputs to models from various input streams

    Dictionary flag governs whether the composition uses the models'
    input dictionaries to compose the input

    If FAST = False, the composer uses input dictionaries

    If FAST = True, the composer uses pre constructed arrays that allow
    vectorization of input mapping. Whenever these need to be updated,
    the build_composer script should be run
    """
    def __init__(self):
        if not os.path.exists(COMPOSER_FILENAME):
            self.build()

    @staticmethod
    def build():
        output = dict()
        fixed = pd.DataFrame.from_dict(LSTG_COLS, 'index')
        fixed.rename(columns=['from'], inplace=True)

        for model_name in model_names.ARRIVAL_MODELS:
        output[model_name] = Composer.build_arrival(model_name, fixed)

    @staticmethod
    def _build_arrival(model_name, fixed):
        """
        Creates input indices for arrival models

        :param model_name: str giving name of the model
        :param fixed: pandas dataframe with colnames as index and
        col position as first and only column ('pos')
        :return: NA
        """
        info_path = '{}/arrival/{}/{}'.format(MODEL_DIR, model_name, INFO_FILENAME)
        info = unpickle(info_path)
        info = pd.DataFrame(data={'to': np.arange(len(info))}, index=info)
        fixed_map = info.merge(fixed, how='inner', left_index=True, right_index=True)
        fixed_map = fixed_map.to_numpy()
        clock_map = info.loc[info.index.isin(ARRIVAL_CLOCK_FEATS), 'to']
        assert len(clock_map) == len(ARRIVAL_CLOCK_FEATS)
        clock_map = clock_map.to_numpy()
        if model_name == model_names.DAYS or model_name == model_names.LOC:
            assert len(clock_map) + fixed_map.shape[0] == len(info)

            return (fixed_map, clock_map)
        elif model_name == model_names.HIST:
            loc_map = info.loc[info.index == 'byr_us', 'pos']
            assert len(clock_map) + fixed_map.shape[0] + 1 == len(info)
        elif model_name == model_names.BIN or model_name == model_names.SEC:
            hist_map = info.loc[info.index == 'byr_hist', 'pos']


