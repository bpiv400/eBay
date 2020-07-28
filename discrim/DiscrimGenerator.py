from rlenv.Composer import Composer
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.generate.Generator import SimulatorGenerator
from rlenv.generate.Recorder import OutcomeRecorder


class DiscrimGenerator(SimulatorGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def generate_composer(self):
        return Composer(cols=self.loader.x_lstg_cols)

    def generate_recorder(self):
        return OutcomeRecorder(verbose=self.verbose,
                               record_sim=False)

    @property
    def env_class(self):
        return SimulatorEnvironment

    def simulate_lstg(self):
        """
        Simulates a particular listing once.
        :return: outcome tuple
        """
        self.environment.reset()
        outcome = self.environment.run()
        return outcome
