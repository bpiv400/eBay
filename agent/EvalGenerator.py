from constants import VALIDATION
from featnames import META
from agent.AgentPlayer import AgentPlayer
from agent.agent_utils import load_agent_params
from rlenv.env_utils import get_env_sim_dir, calculate_slr_gross
from rlenv.simulator.Generator import Generator
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.Composer import AgentComposer


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
        rewards = list()
        for i, lstg in enumerate(self.x_lstg.index):
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            environment = self.setup_env(lstg=lstg, lookup=lookup)

            # update listing in recorder
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)

            # simulate lstg until first sale
            rewards.append(self.simulate_lstg(environment))

    def simulate_lstg(self, environment):
        T = 1
        while True:
            environment.reset()
            sale, price, _ = environment.run()
            if sale:
                return calculate_slr_gross(price=price, list_count=T,
                                           meta=environment.lookup[META])
            else:
                T += 1

    @property
    def records_path(self):
        return None