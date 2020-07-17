from rlenv.generate.Generator import SimulatorGenerator
from rlenv.generate.Recorder import OutcomeRecorder
from agent.util import load_agent_model
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from agent.AgentComposer import AgentComposer
from rlenv.environments.BuyerEnvironment import BuyerEnvironment
from rlenv.environments.SellerEnvironment import SellerEnvironment
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller


class EvalGenerator(SimulatorGenerator):
    def __init__(self, **kwargs):
        super().__init__(verbose=kwargs['verbose'])
        self.agent_params = kwargs['agent_params']
        self.model_kwargs = kwargs['model_kwargs']
        self.run_dir = kwargs['run_dir']
        self.agent = None

    def initialize(self):
        super().initialize()
        self.agent = self.generate_agent()

    @property
    def byr(self):
        return self.agent_params['byr']

    def generate_recorder(self):
        return OutcomeRecorder(verbose=self.verbose,
                               record_sim=True)

    def generate_composer(self):
        return AgentComposer(agent_params=self.agent_params)

    def generate_agent(self):
        model = PgCategoricalAgentModel(**self.model_kwargs)
        model_path = self.run_dir + 'params.pkl'
        load_agent_model(model=model, model_path=model_path)
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
                action = self.agent.con(input_dict=obs)
                agent_tuple = self.environment.step(action)
                done = agent_tuple[2]
            return self.environment.outcome
        else:
            return None
