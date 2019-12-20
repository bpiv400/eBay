from rlenv.env_consts import SIM_COUNT
from rlenv.env_utils import get_chunk_dir
from rlenv.simulator.Generator import Generator
from rlenv.simulator.discrim.DiscrimRecorder import DiscrimRecorder


class DiscrimGenerator(Generator):
    def __init__(self, direct, num, verbose=False):
        super(DiscrimGenerator, self).__init__(direct, num, verbose)
        # initialize recorder
        self.recorder = DiscrimRecorder(self.records_path, self.verbose)


    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        for lstg in self.x_lstg.index:
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            environment = self._setup_env(lstg, lookup)

            # update listing in recorder
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)

            # simulate lstg necessary number of times
            for _ in range(SIM_COUNT):
                self._simulate_lstg(environment)

        # save the recorder
        self.recorder.dump(self.dir, self.recorder_count)


    def _simulate_lstg(self, environment):
        """
        Simulates a particular listing once
        :param environment: RewardEnvironment
        :return: outcome tuple
        """
        environment.reset()
        outcome = environment.run()
        if self.verbose:
            print('Simulation {} concluded'.format(self.recorder.sim))
        return outcome


    @property
    def records_path(self):
        return get_chunk_dir(self.dir, self.chunk, discrim=True)
