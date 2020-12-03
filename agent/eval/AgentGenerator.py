import numpy as np
from agent.AgentComposer import AgentComposer
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from rlenv.generate.Generator import Generator
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller


class AgentGenerator(Generator):
    def __init__(self, model=None, byr=False):
        super().__init__(verbose=False)
        self.model = model
        self.byr = byr

    def generate_composer(self):
        return AgentComposer(byr=self.byr)

    def generate_buyer(self):
        return SimulatedBuyer(full=True)

    def generate_seller(self):
        return SimulatedSeller(full=self.byr)

    @property
    def env_class(self):
        return BuyerEnv if self.byr else SellerEnv

    def simulate_lstg(self):
        obs = self.env.reset()
        if obs is not None:
            done = False
            while not done:
                probs = self.model(observation=obs)
                action = np.argmax(probs)
                obs, _, done, _ = self.env.step(action)
