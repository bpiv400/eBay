import argparse
import torch
from agent.AgentComposer import AgentComposer
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from rlenv.generate.Generator import Generator
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.util import sample_categorical
from agent.models.AgentModel import load_agent_model
from agent.models.HeuristicSlr import HeuristicSlr
from agent.util import get_paths
from rlenv.generate.util import process_sims
from utils import run_func_on_chunks, process_chunk_worker, compose_args
from agent.const import PARAMS
from constants import HOLDOUT_PARTITIONS, DROPOUT_GRID
from featnames import BYR, CON_SET, DROPOUT


class AgentGenerator(Generator):
    def __init__(self, model=None, byr=False, slr=False, con_set=None):
        super().__init__(verbose=False, byr=byr, slr=slr, test=True)
        self.model = model
        self.con_set = con_set
        assert byr or slr

    def generate_composer(self):
        return AgentComposer(byr=self.byr)

    def generate_buyer(self):
        return SimulatedBuyer(full=True)

    def generate_seller(self):
        return SimulatedSeller(full=self.byr)

    @property
    def env_class(self):
        return BuyerEnv if self.byr else SellerEnv

    def generate_env(self):
        return self.env_class(query_strategy=self.query_strategy,
                              loader=self.loader,
                              recorder=self.recorder,
                              verbose=self.verbose,
                              composer=self.composer,
                              con_set=self.con_set,
                              test=True)

    def simulate_lstg(self):
        obs = self.env.reset()
        if obs is not None:
            done = False
            while not done:
                probs = self.model(observation=obs)
                action = int(sample_categorical(probs=probs))
                agent_tuple = self.env.step(action)
                done = agent_tuple[2]
                obs = agent_tuple[0]


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=HOLDOUT_PARTITIONS)
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--suffix', type=str)
    compose_args(arg_dict=PARAMS, parser=parser)
    args = vars(parser.parse_args())

    # convert dropout
    args[DROPOUT] = DROPOUT_GRID[args[DROPOUT]]

    # environment class and run directory
    _, _, run_dir = get_paths(**args)

    # recreate model
    if args['heuristic']:
        if args[BYR]:
            raise NotImplementedError()
        else:
            model = HeuristicSlr()
    else:
        model_args = {k: v for k, v in args.items()
                      if k in [BYR, CON_SET, DROPOUT]}
        model_args['value'] = False
        model = load_agent_model(model_args=model_args, run_dir=run_dir)

    # run in parallel on chunks
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(
            part=args['part'],
            gen_class=AgentGenerator,
            gen_kwargs=dict(
                model=model,
                byr=args[BYR],
                slr=not args[BYR],
                con_set=args[CON_SET]
            )
        )
    )

    # combine and process output
    output_dir = run_dir + '{}/'.format(args['part'])
    if args['heuristic']:
        output_dir += 'heuristic/'
    process_sims(part=args['part'],
                 sims=sims,
                 output_dir=output_dir,
                 byr=args[BYR])


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
