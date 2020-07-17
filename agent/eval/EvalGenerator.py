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
            seller = self.generate_agent()
        return seller

    @property
    def env_class(self):
        if self.byr:
            return BuyerEnvironment
        else:
            return SellerEnvironment

    def _simulate_lstg_slr(self):
        pass

    def _simulate_lstg_byr(self):
        pass

    def simulate_lstg(self):
        """
        Simulates a particular listing either once or until sale.
        :param env: SimulatorEnvironment
        :return: outcome tuple
        """
        if self.byr:
            return self._simulate_lstg_byr()
        else:
            return self._simulate_lstg_slr()