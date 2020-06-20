from agent.AgentPlayer import AgentPlayer
from agent.util import load_agent_model
from agent.AgentComposer import AgentComposer
from sim.outcomes.OutcomeGenerator import OutcomeGenerator
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller


class EvalGenerator(OutcomeGenerator):
    def __init__(self, **kwargs):
        """
        # TODO: The directory strings are out of date
        :param verbose: boolean for whether to print information about threads
        :param model_class: class that inherits agent.models.AgentModel
        :param model_kwargs: dictionary containing kwargs for model_class
        :param str run_dir: path to run directory
        :param int num: chunk number
        :param composer: agent.AgentComposer
        :param record: boolean for whether recorder should dump thread info
        """
        self._composer = kwargs['composer']  # type: AgentComposer
        self.agent_byr = self._composer.byr
        self.delay = self._composer.delay
        self.model_kwargs = kwargs['model_kwargs']
        self.ModelCls = kwargs['model_class']
        self.run_dir = kwargs['run_dir']
        self.path_suffix = kwargs['path_suffix']
        # Todo: why the fuk is part None
        super().__init__(part=None, verbose=kwargs['verbose'])

    def load_chunk(self, chunk=None):
        self.x_lstg, self.lookup = chunk

    def generate_composer(self):
        return self._composer

    def generate_agent(self):
        model = self.ModelCls(**self.model_kwargs)
        model_path = self.run_dir + 'params.pkl'
        load_agent_model(model=model, model_path=model_path)
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

    @property
    def records_path(self):
        return self.run_dir + self.path_suffix
