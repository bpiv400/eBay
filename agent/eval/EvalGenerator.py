from rlenv.generate.Generator import SimulatorGenerator
from agent.AgentComposer import AgentComposer
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.util import sample_categorical


class EvalGenerator(SimulatorGenerator):
    def __init__(self, model=None, env=None, verbose=False):
        super().__init__(verbose=verbose)
        self.env = env
        self.model = model
        self._init_model()

    def _init_model(self):
        for param in self.model.parameters(recurse=True):
            param.requires_grad = False
        self.model.eval()

    def generate_composer(self):
        return AgentComposer(byr=self.model.byr)

    def generate_buyer(self):
        return SimulatedBuyer(full=True)

    def generate_seller(self):
        return SimulatedSeller(full=self.model.byr)

    @property
    def env_class(self):
        return self.env

    def simulate_lstg(self):
        obs = self.environment.reset(next_lstg=False)
        if obs is not None:
            done = False
            while not done:
                probs, _ = self.model(observation=obs)
                action = int(sample_categorical(probs=probs))
                agent_tuple = self.environment.step(action)
                done = agent_tuple[2]
                obs = agent_tuple[0]
