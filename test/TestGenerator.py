from compress_pickle import load
from constants import BYR
from rlenv.Generator import Generator
from test.LstgLog import LstgLog
from test.TestQueryStrategy import TestQueryStrategy
from sim.outcomes.OutcomeRecorder import OutcomeRecorder
from rlenv.util import get_env_sim_subdir
from utils import init_optional_arg


class TestGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(verbose=kwargs['verbose'],
                         part=kwargs['part'])
        init_optional_arg(kwargs=kwargs, name='start',
                          default=None)
        self.start = kwargs['start']
        self.agent = kwargs['agent']
        self.byr = kwargs['byr']
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
        chunk_data = load('{}{}.gz'.format(chunk_dir, chunk))
        self.x_lstg, self.lookup = chunk_data['x_lstg'], chunk_data['lookup']
        self.test_data = load('{}{}_test.gz'.format(chunk_dir, self.chunk))

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
