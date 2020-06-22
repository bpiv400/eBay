import numpy as np
from compress_pickle import load
from agent.util import get_agent_name
from constants import INIT_POLICY_MODELS
from featnames import LSTG
from rlenv.Generator import Generator
from test.LstgLog import LstgLog
from test.TestQueryStrategy import TestQueryStrategy
from sim.outcomes.OutcomeRecorder import OutcomeRecorder
from rlenv.util import get_env_sim_subdir, load_chunk
from utils import init_optional_arg, subset_lstgs


class TestGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(verbose=kwargs['verbose'],
                         part=kwargs['part'])
        init_optional_arg(kwargs=kwargs, name='start',
                          default=None)
        # id of the first lstg in the chunk to simulate
        self.start = kwargs['start']
        # boolean for whether the test is for an agent environment
        self.agent = kwargs['agent']
        # boolean for whether the agent is a byr
        self.byr = kwargs['byr']
        # boolean for whether to the agent selects its delay
        self.delay = kwargs['delay'] or self.byr
        self.test_data = None

    def generate_query_strategy(self):
        pass

    def generate_composer(self):
        pass

    def generate_recorder(self):
        return OutcomeRecorder(records_path="", verbose=self.verbose)

    def load_chunk(self, chunk=None):
        print('Loading test data...')
        chunk_dir = get_env_sim_subdir(part=self.part, chunks=True)
        x_lstg, lookup = load_chunk(input_path='{}{}.gz'.format(chunk_dir, chunk))
        test_data = load('{}{}_test.gz'.format(chunk_dir, chunk))
        test_data = self._remove_extra_models(test_data=test_data)
        # subset inputs to only contain lstgs where the agent has at least 1 action
        if self.agent:
            valid_lstgs = self._get_valid_lstgs(test_data=test_data)
            x_lstg = subset_lstgs(df=x_lstg, lstgs=valid_lstgs)
            lookup = subset_lstgs(df=lookup, lstgs=valid_lstgs)
            test_data['x_thread'] = subset_lstgs(df=test_data['x_thread'], lstgs=valid_lstgs)
            test_data['x_offer'] = subset_lstgs(df=test_data['x_offer'], lstgs=valid_lstgs)
        else:

    def _remove_extra_models(self, test_data):
        """
        Remove the unused policy models from the test data inputs
        """
        unused_models = INIT_POLICY_MODELS
        if self.agent:
            agent_name = get_agent_name(policy=True, byr=self.byr, delay=self.delay)
            unused_models.remove(agent_name)
        for model_name in unused_models:
            del test_data['inputs'][model_name]
        return test_data

    def _get_valid_lstgs(self, test_data=None):
        """
        Retrieves a list of lstgs from the chunk where the agent makes at least one
        turn.

        Verifies this list matches exactly the lstgs with inputs for the relevant model
        :return: pd.Int64Index
        """
        x_offer = test_data['x_offer'].copy()  # x_offer: pd.DataFrame
        if self.byr:
            # all lstgs should have at least 1 action
            lstgs = x_offer.index.get_level_values('lstg').unique()
        else:
            slr_offers = x_offer.index.get_level_values('index') % 2 == 0
            man_offers = x_offer['auto']
            censored_offers = x_offer['censored']
            if self.delay:

                # keep all lstgs with at least 1 thread with at least 1 non-auto seller
                # offer
                keep = np.logical_and(slr_offers, man_offers,
                                      np.logical_not(censored_offers))
                lstgs = x_offer.index.get_level_values('lstg')[keep].unique()
            else:
                # keep all lstgs with at least 1 thread
                # with at least 1 non-auto / non-expiration seller offer
                exp_offers = x_offer['exp']
                keep = np.logical_and(slr_offers, man_offers,
                                      np.logical_not(censored_offers),
                                      np.logical_not(exp_offers))
                lstgs = x_offer.index.get_level_values('lstg')[keep].unique()
        # verify that x_offer based lstgs match lstgs used as model input exactly
        model_name = get_agent_name(policy=True, byr=self.byr, delay=self.delay)
        input_lstgs = test_data['inputs'][model_name][LSTG].index.get_level_values('lstg').unique()
        assert input_lstgs.isin(lstgs).all()
        assert lstgs.isin(input_lstgs).all()
        return lstgs

    def _prune_data(self):
        """
        Remove all lstgs before start, and all lstgs that don't have at least 1 action
        for the agent
        :return:
        """
    def generate(self):
        if self.start is not None:
            start_index = list(self.lookup.index).index(self.start)
            lstgs = self.lookup.index[start_index:]
        else:
            lstgs = self.lookup.index
        for i, lstg in enumerate(lstgs):
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            params = {
                'lstg': lstg,
                'inputs': self.test_data['inputs'],
                'x_thread': self.test_data['x_thread'],
                'x_offer': self.test_data['x_offer'],
                'lookup': lookup
            }

            print('Generating lstg log for {}...'.format(lstg))
            log = LstgLog(params=params)
            print('Log constructed')
            self.query_strategy.update_log(log)
            environment = self.setup_env(lstg=lstg, lookup=lookup)

            # update listing in recorder
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)

            # simulate lstg once
            self.simulate_lstg(environment)

    def simulate_lstg(self, environment):
        """
        Simulates a particular listing once
        :param environment: RewardEnvironment
        :return: outcome tuple
        """
        # simulate once for each buyer in BuyerEnvironment mode
        # this output needs to match first agent action
        environment.reset()
        # going to need to change to return for each agent action
        outcome = environment.run()
        return outcome

    def create_env(self, x_lstg=None, lookup=None):
        # fix this to choose correct environment type
        return TestQueryStrategy(x_lstg=x_lstg, lookup=lookup,
                                 verbose=self.verbose,
                                 query_strategy=self.query_strategy,
                                 composer=self.composer,
                                 recorder=self.recorder)

    @property
    def records_path(self):
        raise RuntimeError("No recorder")
