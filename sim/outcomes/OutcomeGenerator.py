from datetime import datetime as dt
from constants import PARTS_DIR, NO_ARRIVAL_CUTOFF
from featnames import NO_ARRIVAL
from rlenv.Generator import SimulatorGenerator
from rlenv.LstgLoader import ChunkLoader
from rlenv.util import load_chunk, get_env_sim_dir
from sim.outcomes.OutcomeRecorder import OutcomeRecorder


class OutcomeGenerator(SimulatorGenerator):
    def __init__(self, part=None, verbose=False):
        super().__init__(part=part, verbose=verbose)
        self.chunk = None

    def load_chunk(self, chunk=None):
        """
        Returns LstgLoader for the given chunk
        (Subroutine of process_chunk)
        :param chunk: int
        """
        self.chunk = chunk
        base_dir = get_env_sim_dir(part=self.part)
        x_lstg, lookup = load_chunk(base_dir=base_dir,
                                    num=chunk)
        return ChunkLoader(x_lstg=x_lstg, lookup=lookup)

    def generate_recorder(self):
        return OutcomeRecorder(records_path=self.records_path,
                               verbose=self.verbose,
                               record_sim=True)

    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        print('Total listings: {}'.format(len(self.loader)))
        t0, i = dt.now(), 0
        while self.environment.has_next_lstg():
            self.environment.next_lstg()
            # update listing in recorder
            self.recorder.update_lstg(lookup=self.loader.lookup,
                                      lstg=self.loader.lstg)

            # simulate lstg until sale
            self.simulate_lstg()

            # tracking timings
            if (i % 10) == 0 and i != 0:
                print('Avg time per listing: {} seconds'.format(
                    (dt.now() - t0).total_seconds() / (i+1)))
            i += 1

        # save the recorder
        self.recorder.dump()

    def simulate_lstg(self):
        """
        Simulates a particular listing either once or until sale.
        :return: outcome tuple
        """
        # TODO: Replace after Etan response
        # sim_once = self.loader.lookup[NO_ARRIVAL] > NO_ARRIVAL_CUTOFF
        sim_once = True
        while True:
            self.environment.reset()
            outcome = self.environment.run()
            if outcome.sale or sim_once:
                return outcome

    @property
    def records_path(self):
        return PARTS_DIR + '{}/outcomes/{}.gz'.format(self.part, self.chunk)
