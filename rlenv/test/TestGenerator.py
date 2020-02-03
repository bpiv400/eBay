from compress_pickle import load
from rlenv.simulator.Generator import Generator
from rlenv.test.LstgLog import LstgLog
from rlenv.test.TestEnvironment import TestEnvironment
from rlenv.simulator.discrim.DiscrimRecorder import DiscrimRecorder
from rlenv.env_utils import get_env_sim_subdir


class TestGenerator(Generator):
    def __init__(self, direct, num, verbose=False):
        super().__init__(direct, num, verbose)
        chunk_dir = get_env_sim_subdir(base_dir=direct, chunks=True)
        print('Loading test inputs...')
        self.test_data = load('{}{}_test.gz'.format(chunk_dir, num))
        self.recorder = DiscrimRecorder("", self.verbose)

    def generate(self):
        for i, lstg in enumerate(self.x_lstg.index):
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
            print('Generating lstg log...')
            log = LstgLog(params=params)
            print('Log constructed')
            environment = self.setup_env(lstg=lstg, lookup=lookup, log=log)

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
        environment.reset()
        outcome = environment.run()
        return outcome

    def create_env(self, x_lstg=None, lookup=None, log=None):
        return TestEnvironment(buyer=self.buyer, seller=self.seller,
                               arrival=self.arrival, x_lstg=x_lstg,
                               lookup=lookup, verbose=self.verbose,
                               log=log, composer=self.composer,
                               recorder=self.recorder)

    @property
    def records_path(self):
        raise RuntimeError("No recorder")
