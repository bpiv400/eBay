from rlenv.Composer import Composer
from rlenv.EBayEnv import EBayEnv
from rlenv.generate.Generator import Generator
from rlenv.generate.util import process_sims
from utils import run_func_on_chunks, process_chunk_worker, input_partition
from constants import PARTS_DIR


class SimulatorEnv(EBayEnv):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def run(self):
        """
        Runs a simulation of a single lstg until sale or expiration

        :return: a 3-tuple of (bool, float, int) giving whether the listing sells,
        the amount it sells for if it sells, and the amount of time it took to sell
        """
        super().run()
        return self.outcome

    def is_agent_turn(self, event):
        return False


class OutcomeGenerator(Generator):

    def generate_composer(self):
        return Composer(cols=self.loader.x_lstg_cols)

    @property
    def env_class(self):
        return SimulatorEnv

    def simulate_lstg(self):
        """
        Simulates listing once.
        :return: None
        """
        self.env.reset()
        self.env.run()


def main():
    part = input_partition()

    # process chunks in parallel
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(part=part, gen_class=OutcomeGenerator)
    )

    # concatenate, clean, and save
    output_dir = PARTS_DIR + '{}/sim/'.format(part)
    process_sims(part=part, sims=sims, output_dir=output_dir)


if __name__ == '__main__':
    main()
