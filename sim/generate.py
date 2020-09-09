import argparse
from rlenv.Composer import Composer
from rlenv.generate.Generator import Generator
from rlenv.generate.util import process_sims
from sim.envs import SimulatorEnv, NoSlrExpEnv, SlrAccRejEnv
from utils import run_func_on_chunks, process_chunk_worker
from constants import PARTS_DIR, AGENT_PARTITIONS


class OutcomeGenerator(Generator):

    def __init__(self, env):
        super().__init__(verbose=False, test=True)
        self.env = env

    def generate_composer(self):
        return Composer(cols=self.loader.x_lstg_cols)

    @property
    def env_class(self):
        return self.env

    def simulate_lstg(self):
        """
        Simulates listing once.
        :return: None
        """
        self.env.reset()
        self.env.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=AGENT_PARTITIONS)
    parser.add_argument('--noslrexp', action='store_true')
    parser.add_argument('--slraccrej', action='store_true')
    args = parser.parse_args()

    # environment and output folder
    if args.noslrexp:
        env = NoSlrExpEnv
        output_dir = PARTS_DIR + '{}/noslrexp/'.format(args.part)
    elif args.slraccrej:
        env = SlrAccRejEnv
        output_dir = PARTS_DIR + '{}/slraccrej/'.format(args.part)
    else:
        env = SimulatorEnv
        output_dir = PARTS_DIR + '{}/sim/'.format(args.part)

    # process chunks in parallel
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(part=args.part,
                         gen_class=OutcomeGenerator,
                         gen_kwargs=dict(env=env))
    )

    # concatenate, clean, and save
    process_sims(part=args.part, sims=sims, output_dir=output_dir)


if __name__ == '__main__':
    main()
