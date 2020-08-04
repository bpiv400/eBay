from rlenv.generate.Generator import SimulatorGenerator
from agent.AgentComposer import AgentComposer
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from agent.models.BetaCategorical import BetaCategorical
from agent.const import NUM_ACTIONS_SLR, NUM_ACTIONS_BYR
from agent.util import pack_dist_info


class EvalGenerator(SimulatorGenerator):
    def __init__(self, model=None, env=None, verbose=False):
        super().__init__(verbose=verbose)
        self.env = env
        self.model = model
        for param in self.model.parameters(recurse=True):
            param.requires_grad = False
        self.model.eval()
        num_actions = NUM_ACTIONS_BYR if self.model.byr else NUM_ACTIONS_SLR
        self.distribution = BetaCategorical(dim=num_actions)

    def generate_composer(self):
        return AgentComposer(byr=self.model.byr)

    def generate_buyer(self):
        return SimulatedBuyer(full=True)

    def generate_seller(self):
        if self.model.byr:
            seller = SimulatedSeller(full=True)
        else:
            seller = SimulatedSeller(full=False)
        return seller

    @property
    def env_class(self):
        return self.env

    def simulate_lstg(self):
        obs = self.environment.reset(next_lstg=False)
        if obs is not None:
            done = False
            while not done:
                params, _ = self.model(obs=obs)
                dist_info = pack_dist_info(params)
                action = self.distribution.sample(dist_info)
                agent_tuple = self.environment.step(action)
                done = agent_tuple[2]
                obs = agent_tuple[0]
            return self.environment.outcome
        else:
            return None
