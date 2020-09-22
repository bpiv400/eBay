import argparse
import torch
from agent.AgentComposer import AgentComposer
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from rlenv.generate.Generator import Generator
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.util import sample_categorical
from agent.models.AgentModel import AgentModel
from agent.models.HeuristicSlr import HeuristicSlr
from agent.util import get_paths
from rlenv.generate.util import process_sims
from utils import run_func_on_chunks, process_chunk_worker, compose_args
from agent.const import PARAMS, AGENT_STATE, OPTIM_STATE
from constants import VALIDATION, TRAIN_RL, TRAIN_RL, TEST


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


def load_agent_model(model_args=None, run_dir=None):
    model = AgentModel(**model_args)
    path = run_dir + 'params.pkl'
    d = torch.load(path, map_location=torch.device('cpu'))
    if OPTIM_STATE in d:
        d = d[AGENT_STATE]
        torch.save(d, path)
    d = {k: v for k, v in d.items() if not k.startswith('value')}
    model.load_state_dict(d, strict=True)
    for param in model.parameters(recurse=True):
        param.requires_grad = False
    model.eval()
    return model


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=[TRAIN_RL, TRAIN_RL, VALIDATION, TEST])
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--suffix', type=str)
    compose_args(arg_dict=PARAMS, parser=parser)
    args = parser.parse_args()

    # error checking
    if args.byr:
        assert args.part != TRAIN_RL

    # environment class and run directory
    _, _, run_dir = get_paths(**vars(args))

    # recreate model
    if args.heuristic:
        if args.byr:
            raise NotImplementedError()
        else:
            model = HeuristicSlr()
    else:
        model = load_agent_model(
            model_args=dict(byr=args.byr, con_set=args.con_set, value=False),
            run_dir=run_dir
        )

    # run in parallel on chunks
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(
            part=args.part,
            gen_class=AgentGenerator,
            gen_kwargs=dict(
                model=model,
                byr=args.byr,
                slr=not args.byr,
                con_set=args.con_set
            )
        )
    )

    # combine and process output
    output_dir = run_dir + '{}/'.format(args.part)
    if args.heuristic:
        output_dir += 'heuristic/'
    process_sims(part=args.part, sims=sims, output_dir=output_dir)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
