import os
import numpy as np
from agent.AgentComposer import AgentComposer
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from agent.models.AgentModel import load_agent_model
from agent.models.heuristics import HeuristicSlr, HeuristicByr
from agent.eval.util import sim_args
from agent.util import get_run_dir, get_sim_dir
from rlenv.generate.Generator import OutcomeGenerator
from rlenv.generate.Recorder import OutcomeRecorder
from rlenv.Player import SimulatedSeller
from utils import topickle
from constants import OUTCOME_SIMS
from featnames import DELTA, TURN_COST


class AgentGenerator(OutcomeGenerator):
    def __init__(self, model=None, byr=None, delta=None, turn_cost=None):
        super().__init__(verbose=False, test=False)
        self.model = model
        self.byr = byr
        self.delta = delta
        self.turn_cost = turn_cost

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
        args[TURN_COST] = self.turn_cost
        return self.env_class(**args)

    @property
    def env_class(self):
        return BuyerEnv if self.byr else SellerEnv


def main():
    args = sim_args(num=True)

    # output directory
    params = {k: v for k, v in vars(args).items() if k != 'num'}
    output_dir = get_sim_dir(**params)
    outcome_dir = output_dir + 'outcomes/'

    # create output folder
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
        model_cls = HeuristicByr if args.byr else HeuristicSlr
        model = model_cls(delta=args.delta)
    else:
        run_dir = get_run_dir(byr=args.byr,
                              delta=args.delta,
                              turn_cost=args.turn_cost)
        model_args = dict(byr=args.byr, value=False)
        model = load_agent_model(model_args=model_args, run_dir=run_dir)

    # generator
    gen = AgentGenerator(model=model, byr=args.byr,
                         delta=args.delta, turn_cost=args.turn_cost)

    # process one chunk
    df = gen.process_chunk(part=args.part, chunk=chunk, num_sims=OUTCOME_SIMS)

    # save
    topickle(df, path)


if __name__ == '__main__':
    main()
