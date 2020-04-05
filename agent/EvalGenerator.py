from agent.AgentPlayer import AgentPlayer
from agent.agent_utils import load_agent_params
from rlenv.simulator.Generator import Generator
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.Composer import AgentComposer
from rlenv.env_utils import get_env_sim_dir
from constants import VALIDATION


class EvalGenerator(Generator):
    def __init__(self, **kwargs):
        """
        :param num
        :param verbose
        :param model_class
        :param model_kwargs
        :param run_dir
        :param composer
        """
        self._composer = kwargs['composer']  # type: AgentComposer
        self.agent_byr = self._composer.byr
        self.delay = self._composer.delay
        self.model_kwargs = kwargs['model_kwargs']
        self.ModelCls = kwargs['model_class']
        self.run_dir = kwargs['run_dir']
        super().__init__(get_env_sim_dir(VALIDATION), kwargs['num'],
                         verbose=kwargs['verbose'])

    def generate_composer(self):
        return self._composer

    def generate_agent(self):
        model = self.ModelCls(**self.model_kwargs)
        load_agent_params(model=model, run_dir=self.run_dir)
        agent = AgentPlayer(agent_model=model)
        return agent

    def generate_buyer(self):
        if self.agent_byr:
            buyer = self.generate_agent()
        else:
            buyer = SimulatedBuyer(full=True)
        return buyer

    def generate_seller(self):
        if self.agent_byr:
            seller = SimulatedSeller(full=True)
        else:
            seller = self.generate_agent()
        return seller

    def generate(self):
        pass

    def simulate_lstg(self, environment):
        pass

    @property
    def records_path(self):
        pass