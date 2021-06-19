import os
import numpy as np
import torch
from agent.AgentComposer import AgentComposer
from agent.AgentModel import AgentModel
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from agent.heuristics.HeuristicSlr import HeuristicSlr
from agent.heuristics.HeuristicByr import HeuristicByr
from agent.eval.util import sim_args
from agent.util import get_run_dir, get_sim_dir
from rlenv.generate.Generator import OutcomeGenerator
from rlenv.generate.Recorder import OutcomeRecorder
from rlenv.Player import SimulatedSeller
from utils import topickle
from constants import OUTCOME_SIMS
from featnames import DELTA, X_THREAD


class AgentGenerator(OutcomeGenerator):
    def __init__(self, model=None, byr=None, delta=None):
        super().__init__(verbose=False, test=False)
        self.model = model
        self.byr = byr
        self.delta = delta

    def simulate_lstg(self):
        obs = self.env.reset()
        if obs is not None:
            done = False
            while not done:
                probs = self.model(observation=obs)
                action = np.argmax(probs)
                obs, _, done, _ = self.env.step(action)

    def generate_recorder(self):
        return OutcomeRecorder(verbose=self.verbose, byr=self.byr)

    def generate_composer(self):
        return AgentComposer(byr=self.byr)

    def generate_seller(self):
        return SimulatedSeller(full=self.byr)

    def generate_env(self):
        args = self.env_args
        args[DELTA] = self.delta
        return self.env_class(**args)

    @property
    def env_class(self):
        return BuyerEnv if self.byr else SellerEnv


def load_agent_model(args=None):
    model = AgentModel(byr=args.byr, value=False)
    run_dir = get_run_dir(byr=args.byr,
                          delta=args.delta,
                          turn_cost=args.turn_cost)
    path = run_dir + 'params.pkl'
    d = torch.load(path, map_location=torch.device('cpu'))
    if 'agent_state_dict' in d:
        d = d['agent_state_dict']
    d = {k: v for k, v in d.items() if not k.startswith('value')}
    model.load_state_dict(d, strict=True)
    for param in model.parameters(recurse=True):
        param.requires_grad = False
    model.eval()
    return model


def main():
    args = sim_args(num=True)

    # output directory
    params = {k: v for k, v in vars(args).items() if k != 'num'}
    sim_dir = get_sim_dir(**params)

    # check for collated output
    if os.path.isfile(sim_dir + '{}.pkl'.format(X_THREAD)):
        print('Output already exists.')
        exit(0)

    # create output folder
    outcome_dir = sim_dir + 'outcomes/'
    if not os.path.isdir(outcome_dir):
        os.makedirs(outcome_dir)

    # check if chunk has already been processed
    chunk = args.num - 1
    path = outcome_dir + '{}.pkl'.format(chunk)
    if os.path.isfile(path):
        print('Chunk {} already exists.'.format(chunk))
        exit(0)

    # model
    if args.heuristic:
        if args.byr:
            args.delta = 1
            model = HeuristicByr(index=args.index)
        else:
            model = HeuristicSlr(delta=args.delta)
        num_sims = 1
    else:
        model = load_agent_model(args)
        num_sims = OUTCOME_SIMS

    # generator
    gen = AgentGenerator(model=model, byr=args.byr, delta=args.delta)

    # process one chunk
    df = gen.process_chunk(part=args.part, chunk=chunk, num_sims=num_sims)

    # save
    topickle(df, path)


if __name__ == '__main__':
    main()
