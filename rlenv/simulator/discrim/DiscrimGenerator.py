from rlenv.env_consts import SIM_DISCRIM_DIR
from rlenv.simulator.Generator import Generator
from rlenv.simulator.discrim.DiscrimRecorder import DiscrimRecorder
from datetime import datetime as dt


class DiscrimGenerator(Generator):
    def __init__(self, direct, num, verbose=False):
        super(DiscrimGenerator, self).__init__(direct, num, verbose)
        # initialize recorder
        self.recorder = DiscrimRecorder(self.records_path, self.verbose)

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
            environment = self._setup_env(lstg, lookup)

            # update listing in recorder
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)

            # simulate lstg once
            self._simulate_lstg(environment)

            if (i % 10) == 0 and i != 0:
                print('Avg time per listing: {} seconds'.format(
                    (dt.now() - t0).total_seconds() / (i+1)))

        # save the recorder
        self.recorder.dump()

    def _simulate_lstg(self, environment):
        """
        Simulates a particular listing once
        :param environment: RewardEnvironment
        :return: outcome tuple
        """
        environment.reset()
        outcome = environment.run()
        return outcome

    @property
    def records_path(self):
        return '{}{}/{}.gz'.format(self.dir, SIM_DISCRIM_DIR, self.chunk)
