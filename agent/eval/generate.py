import argparse
import torch
from agent.AgentComposer import AgentComposer
from agent.models.HeuristicSlr import HeuristicSlr
from agent.util import get_paths, load_agent_model
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from rlenv.generate.Generator import Generator
from rlenv.generate.util import process_sims
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.util import sample_categorical
from utils import run_func_on_chunks, process_chunk_worker
from constants import VALIDATION, RL_SLR, RL_BYR, TEST


class AgentGenerator(Generator):
    def __init__(self, model=None, byr_agent=None, verbose=False):
        super().__init__(verbose=verbose, byr_agent=byr_agent)
        self.model = model

    def generate_composer(self):
        return AgentComposer(byr=self.model.byr)

    def generate_buyer(self):
        return SimulatedBuyer(full=True)

    def generate_seller(self):
        return SimulatedSeller(full=self.model.byr)

    @property
    def env_class(self):
        return BuyerEnv if self.model.byr else SellerEnv

    def simulate_lstg(self):
        obs = self.env.reset(next_lstg=False)
        if obs is not None:
            done = False
            while not done:
                probs, _ = self.model(observation=obs)
                action = int(sample_categorical(probs=probs))
                agent_tuple = self.env.step(action)
                done = agent_tuple[2]
                obs = agent_tuple[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--part', choices=[RL_SLR, RL_BYR, VALIDATION, TEST])
    parser.add_argument('--name', type=str)
    parser.add_argument('--heuristic', action='store_true')
    args = parser.parse_args()

    # environment class and run directory
    _, _, run_dir = get_paths(byr=args.byr, name=args.name)

    # recreate model
    if args.heuristic:
        if args.byr:
            raise NotImplementedError()
        else:
            model = HeuristicSlr()
    else:
        nocon = 'nocon' in args.name
        model = load_agent_model(
            model_args=dict(byr=args.byr, nocon=nocon),
            run_dir=run_dir
        )

    # run in parallel on chunks
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(
            part=args.part,
            gen_class=AgentGenerator,
            gen_kwargs=dict(model=model, byr_agent=args.byr)
        )
    )

    # combine and process output
    output_dir = run_dir + '{}/'.format(args.part)
    if args.heuristic:
        output_dir += 'heuristic/'
    process_sims(part=args.part, sims=sims, output_dir=output_dir)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
