from rlenv.Generator import SimulatorGenerator
from sim.outcomes.OutcomeRecorder import OutcomeRecorder
from datetime import datetime as dt


class OutcomeGenerator(SimulatorGenerator):
    def __init__(self, direct=None, verbose=False, start=None):
        super().__init__(direct=direct, verbose=verbose)

    def generate_recorder(self):
        return OutcomeRecorder(records_path=self.records_path,
                               verbose=self.verbose,
                               record_sim=True)

    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        print('Total listings: {}'.format(len(self.x_lstg)))
        t0 = dt.now()
        for i, lstg in enumerate(self.x_lstg.index):
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            environment = self.setup_env(lstg=lstg, lookup=lookup)

            # update listing in recorder
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)

            # simulate lstg until sale
            self.simulate_lstg(environment)

            if (i % 10) == 0 and i != 0:
                print('Avg time per listing: {} seconds'.format(
                    (dt.now() - t0).total_seconds() / (i+1)))

        # save the recorder
        self.recorder.dump()

    def simulate_lstg(self, environment):
        """
        Simulates a particular listing once
        :param environment: RewardEnvironment
        :return: outcome tuple
        """
        while True:
            environment.reset()
            outcome = environment.run()
            if outcome.sale:
                return outcome

    @property
    def records_path(self):
        return self.dir + '{}.gz'.format(self.chunk)
