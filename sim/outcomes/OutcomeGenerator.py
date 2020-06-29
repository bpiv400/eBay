from rlenv.Generator import SimulatorGenerator
from rlenv.util import get_env_sim_subdir
from sim.outcomes.OutcomeRecorder import OutcomeRecorder
from datetime import datetime as dt
from constants import NO_ARRIVAL_CUTOFF
from featnames import NO_ARRIVAL


class OutcomeGenerator(SimulatorGenerator):
    def __init__(self, part=None, verbose=False):
        super().__init__(part=part, verbose=verbose)

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

    def simulate_lstg(self, env):
        """
        Simulates a particular listing either once or until sale.
        :param env: SimulatorEnvironment
        :return: outcome tuple
        """
        sim_once = env.lookup[NO_ARRIVAL] > NO_ARRIVAL_CUTOFF
        while True:
            env.reset()
            outcome = env.run()
            if outcome.sale or sim_once:
                return outcome

    @property
    def records_path(self):
        out_dir = get_env_sim_subdir(part=self.part,
                                     discrim=True)
        return out_dir + '{}.gz'.format(self.chunk)
