import torch
from rlenv.generate.Generator import SimulatorGenerator
from rlenv.generate.Recorder import OutcomeRecorder
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from agent.AgentComposer import AgentComposer
from rlenv.environments.BuyerEnvironment import BuyerEnvironment
from rlenv.environments.SellerEnvironment import SellerEnvironment
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller


class EvalGenerator(SimulatorGenerator):
    def __init__(self, byr=None, dropout=None, run_dir=None, verbose=False):
        super().__init__(verbose=verbose)
        self.byr = byr
        self.dropout = dropout
        self.run_dir = run_dir
        self.agent = None

    def initialize(self):
        super().initialize()
        self.agent = self.generate_agent()

    def generate_recorder(self):
        return OutcomeRecorder(verbose=self.verbose,
                               record_sim=True)

    def generate_composer(self):
        return AgentComposer(byr=self.byr)

    def generate_agent(self):
        model = PgCategoricalAgentModel(byr=self.byr, dropout=self.dropout)
        state_dict = torch.load(self.run_dir + 'params.pkl',
                                map_location=torch.device('cpu'))
        model.load_state_dict(state_dict=state_dict, strict=True)
        for param in model.parameters(recurse=True):
            param.requires_grad = False
        model.eval()
        return model

    def generate_buyer(self):
        return SimulatedBuyer(full=True)

    def generate_seller(self):
        if self.byr:
            seller = SimulatedSeller(full=True)
        else:
            seller = SimulatedSeller(full=False)
        return seller

    @property
    def env_class(self):
        if self.byr:
            return BuyerEnvironment
        else:
            return SellerEnvironment

    def simulate_lstg(self):
        obs = self.environment.reset(next_lstg=False)
        if obs is not None:
            done = False
            while not done:
                action = self.agent.con(obs=obs)
                agent_tuple = self.environment.step(action)
                done = agent_tuple[2]
                obs = agent_tuple[0]
            return self.environment.outcome
        else:
            return None
