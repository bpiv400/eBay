import numpy as np
from agent.AgentComposer import AgentComposer
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from rlenv.generate.Generator import OutcomeGenerator
from rlenv.interfaces.PlayerInterface import SimulatedSeller


class AgentGenerator(OutcomeGenerator):
    def __init__(self, model=None):
        super().__init__(verbose=False, test=False)
        self.model = model

    def simulate_lstg(self):
        obs = self.env.reset()
        if obs is not None:
            done = False
            while not done:
                probs = self.model(observation=obs)
                action = np.argmax(probs)
                obs, _, done, _ = self.env.step(action)


class SellerGenerator(AgentGenerator):

    def generate_composer(self):
        return AgentComposer(byr=False)

    def generate_seller(self):
        return SimulatedSeller(full=False)

    @property
    def env_class(self):
        return SellerEnv


class BuyerGenerator(AgentGenerator):
    def __init__(self, model=None, agent_thread=1):
        self.agent_thread = agent_thread
        super().__init__(model=model)

    def generate_composer(self):
        return AgentComposer(byr=True)

    def generate_seller(self):
        return SimulatedSeller(full=True)

    @property
    def env_class(self):
        return BuyerEnv

    def generate_env(self):
        env_args = self.env_args.copy()
        env_args['agent_thread'] = self.agent_thread
        return self.env_class(**env_args)
