import numpy as np
from compress_pickle import load
from agent.const import BYR, FEAT_TYPE, CON_TYPE, FULL_CON
from agent.util import get_agent_name
from agent.AgentComposer import AgentComposer
from constants import INIT_POLICY_MODELS
from featnames import LSTG, DELAY, BYR_HIST
from rlenv.Composer import Composer
from rlenv.environments.BuyerEnvironment import BuyerEnvironment
from rlenv.environments.SellerEnvironment import SellerEnvironment
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.Generator import Generator
from rlenv.util import get_env_sim_subdir, load_chunk
from sim.outcomes.OutcomeRecorder import OutcomeRecorder
from test.LstgLog import LstgLog
from test.TestQueryStrategy import TestQueryStrategy
from test.util import subset_inputs
from test.TestLoader import TestLoader
from utils import init_optional_arg, subset_lstgs


class TestGenerator(Generator):
    def __init__(self, **kwargs):
        super().__init__(verbose=kwargs['verbose'],
                         part=kwargs['part'])
        init_optional_arg(kwargs=kwargs, name='start',
                          default=None)
        # id of the first lstg in the chunk to simulate
        self.start = kwargs['start']
        # boolean for whether the test is for an agent environment
        self.agent = kwargs['agent']
        # boolean for whether the agent is a byr
        self.byr = kwargs['role'] == BYR
        # feat type
        self.feat_type = kwargs[FEAT_TYPE]
        # boolean for whether to the agent selects its delay
        self.delay = kwargs[DELAY] or self.byr
        self.test_data = None

    def generate_query_strategy(self):
        return TestQueryStrategy()

    def generate_composer(self):
        if self.agent:
            agent_params = {
                BYR: self.byr,
                DELAY: self.delay,
                FEAT_TYPE: self.feat_type,
                CON_TYPE: FULL_CON,
                BYR_HIST: None
            }
            return AgentComposer(cols=self.loader.x_lstg_cols,
                                 agent_params=agent_params)
        else:
            return Composer(cols=self.loader.x_lstg_cols)

    def generate_recorder(self):
        return OutcomeRecorder(records_path="", verbose=self.verbose)

    def load_chunk(self, chunk=None):
        print('Loading test data...')
        chunk_dir = get_env_sim_subdir(part=self.part, chunks=True)
        x_lstg, lookup = load_chunk(input_path='{}{}.gz'.format(chunk_dir, chunk))
        test_data = load('{}{}_test.gz'.format(chunk_dir, chunk))
        test_data = self._remove_extra_models(test_data=test_data)
        # subset inputs to only contain lstgs where the agent has at least 1 action
        valid_lstgs = self._get_valid_lstgs(test_data=test_data, lookup=lookup)
        if len(valid_lstgs) < len(lookup.index):
            x_lstg = subset_lstgs(df=x_lstg, lstgs=valid_lstgs)
            lookup = subset_lstgs(df=lookup, lstgs=valid_lstgs)
            test_data['x_thread'] = subset_lstgs(df=test_data['x_thread'], lstgs=valid_lstgs)
            test_data['x_offer'] = subset_lstgs(df=test_data['x_offer'], lstgs=valid_lstgs)
            test_data['inputs'] = subset_inputs(input_data=test_data,
                                                value=valid_lstgs, level='lstg')
        return TestLoader(x_lstg=x_lstg, lookup=lookup, test_data=test_data)

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

    def _get_valid_lstgs(self, test_data=None, lookup=None):
        """
        Retrieves a list of lstgs from the chunk where the agent makes at least one
        turn.

        Verifies this list matches exactly the lstgs with inputs for the relevant model
        :return: pd.Int64Index
        """
        if self.start is not None:
            start_index = list(lookup.index).index(self.start)
            start_lstgs = lookup.index[start_index:]
        else:
            start_lstgs = lookup.index
        if self.agent:
            agent_lstgs = self._get_agent_lstgs(test_data=test_data)
            lstgs = start_lstgs.intersect1d(agent_lstgs,
                                            start_lstgs)
            return lstgs
        else:
            return start_lstgs

    def _get_agent_lstgs(self, test_data=None):
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

    def generate(self):
        while self.environment.has_next_lstg():
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

    @property
    def env_class(self):
        if self.agent:
            if self.byr:
                env_class = BuyerEnvironment
            else:
                env_class = SellerEnvironment
        else:
            env_class = SimulatorEnvironment
        return env_class

    @property
    def records_path(self):
        raise RuntimeError("No recorder")
