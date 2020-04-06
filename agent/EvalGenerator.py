from constants import RL_EVAL_DIR
from featnames import META
from agent.AgentPlayer import AgentPlayer
from agent.agent_utils import load_agent_params
from agent.AgentComposer import AgentComposer
from rlenv.env_utils import calculate_slr_gross, load_chunk
from rlenv.simulator.Generator import Generator
from rlenv.simulator.discrim.DiscrimRecorder import DiscrimRecorder
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller


class EvalGenerator(Generator):
    def __init__(self, **kwargs):
        """
        :param itr
        :param verbose
        :param model_class
        :param model_kwargs
        :param run_dir
        :param composer
        :param record
        """
        self.itr = kwargs['itr']
        self._composer = kwargs['composer']  # type: AgentComposer
        self.agent_byr = self._composer.byr
        self.delay = self._composer.delay
        self.model_kwargs = kwargs['model_kwargs']
        self.ModelCls = kwargs['model_class']
        self.run_dir = kwargs['run_dir']
        self.record = kwargs['record']
        super().__init__(direct=None,
                         verbose=kwargs['verbose'])

    def load_chunk(self, chunk=None):
        path = '{}{}.gz'.format(RL_EVAL_DIR, chunk)
        self.x_lstg, self.lookup = load_chunk(input_path=path)

    def generate_recorder(self):
        return DiscrimRecorder(verbose=self.verbose, records_path=self.records_path,
                               record_sim=True)

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
        if self.record:
            self.recorder.dump()
            self.recorder.reset_recorders()

        return rewards

    def simulate_lstg(self, environment):
        list_count = 1
        while True:
            environment.reset()
            sale, price, _ = environment.run()
            if sale:
                return calculate_slr_gross(price=price, list_count=list_count,
                                           meta=environment.lookup[META])
            else:
                list_count += 1

    @property
    def records_path(self):
        return '{}{}_{}.gz'.format(self.run_dir, self.itr, self.chunk)
