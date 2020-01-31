import pandas as pd
from compress_pickle import load
from rlenv.simulator.Generator import Generator
from rlenv.test.LstgLog import LstgLog
from rlenv.test.TestEnvironment import TestEnvironment
from rlenv.env_utils import get_env_sim_subdir
from rlenv.env_consts import MODELS


class TestGenerator(Generator):
    def __init__(self, direct, num, verbose=False):
        super().__init__(direct, num, verbose)
        chunk_dir = get_env_sim_subdir(base_dir=direct, chunks=True)
        self.test_data = load('{}{}_test.gz'.format(chunk_dir, num))
        self.log = None

    def generate(self):
        for i, lstg in enumerate(self.x_lstg.index):
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            environment = self.setup_env(lstg, lookup)

            # simulate lstg once
            self.simulate_lstg(environment)

    def setup_env(self, lstg, lookup):
        params = {
            'lstg': lstg,
            'inputs': self.subset_inputs(lstg),
            'x_thread': TestGenerator.subset_df(df=self.test_data['x_thread'],
                                                lstg=lstg),
            'x_offer': TestGenerator.subset_df(df=self.test_data['x_offer'],
                                               lstg=lstg)
        }
        self.log = LstgLog(params=params)
        return super().setup_env(lstg=lstg, lookup=lookup)

    def simulate_lstg(self, environment):
        """
        Simulates a particular listing once
        :param environment: RewardEnvironment
        :return: outcome tuple
        """
        environment.reset()
        outcome = environment.run()
        return outcome

    def create_env(self, x_lstg=None, lookup=None):
        return TestEnvironment(buyer=self.buyer, seller=self.seller,
                               arrival=self.arrival, x_lstg=x_lstg,
                               lookup=lookup, verbose=self.verbose,
                               log=self.log)

    def subset_inputs(self, lstg):
        inputs = dict()
        input_data = self.test_data['inputs']
        for model in MODELS:
            inputs[model] = dict()
            index_is_cached = False
            curr_index = None
            for input_group, feats_df in input_data[model].items():
                if not index_is_cached:
                    if lstg in feats_df.index.unique(level='lstg'):
                        full_lstgs = feats_df.index.get_level_values('lstg')
                        curr_index = full_lstgs == lstg
                    else:
                        curr_index = None
                    index_is_cached = True
                if curr_index is None:
                    subset = feats_df.loc[curr_index, :]
                else:
                    subset = None
                inputs[model][input_group] = subset
        return input_data

    @staticmethod
    def subset_df(df=None, lstg=None):
        """
        Subsets an arbitrary dataframe to only contain rows for the given
        lstg
        :param df: pd.DataFrame
        :param lstg: integer giving lstg
        :return: pd.Series or pd.DataFrame
        """
        if isinstance(df.index, pd.MultiIndex):
            lstgs = df.index.unique(level='lstg')
            if lstg in lstgs:
                return df.xs(lstg, level='lstg', drop_level=True)
        else:
            if lstg in df.index:
                return df.loc[lstg, :]
        return None

    @property
    def records_path(self):
        raise RuntimeError("No recorder")
