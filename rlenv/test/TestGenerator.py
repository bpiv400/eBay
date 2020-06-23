from rlenv.Generator import SimulatorGenerator
from rlenv.test.LstgLog import LstgLog
from rlenv.test.TestEnvironment import TestEnvironment
from rlenv.test.util import load_all_inputs, load_reindex
from sim.outcomes.OutcomeRecorder import OutcomeRecorder


class TestGenerator(SimulatorGenerator):
    def __init__(self, part=None, verbose=False):
        super().__init__(part=part, verbose=verbose)
        self.data = None

    def generate_recorder(self):
        return OutcomeRecorder(records_path="",
                               verbose=self.verbose)

    def load_chunk(self, chunk=None):
        super().load_chunk(chunk=chunk)
        lstgs = self.lookup.index
        self.data = dict()
        self.data['inputs'] = load_all_inputs(part=self.part,
                                              lstgs=lstgs)
        for name in ['x_thread', 'x_offer']:
            self.data[name] = load_reindex(part=self.part,
                                           name=name,
                                           lstgs=lstgs)

    def generate(self):
        lstgs = self.lookup.index
        for i, lstg in enumerate(lstgs):
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            params = {
                'lstg': lstg,
                'inputs': self.data['inputs'],
                'x_thread': self.data['x_thread'],
                'x_offer': self.data['x_offer'],
                'lookup': lookup
            }
            print('Generating lstg log for {}...'.format(lstg))
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
