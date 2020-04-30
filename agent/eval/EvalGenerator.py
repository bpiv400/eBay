from featnames import META, START_PRICE
from agent.AgentPlayer import AgentPlayer
from agent.agent_utils import load_agent_model
from agent.AgentComposer import AgentComposer
from rlenv.simulator.Generator import Generator
from rlenv.simulator.discrim.DiscrimRecorder import DiscrimRecorder
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from utils import slr_reward
from processing.processing_consts import MONTHLY_DISCOUNT


class EvalGenerator(Generator):
    def __init__(self, **kwargs):
        """
        :param verbose: boolean for whether to print information about threads
        :param model_class: class that inherits agent.models.AgentModel
        :param model_kwargs: dictionary containing kwargs for model_class
        :param str run_dir: path to run directory
        :param int itr: model iteration
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
        self.itr = kwargs['itr']
        self.num = kwargs['num']
        self.record = kwargs['record']
        super().__init__(direct=None,
                         verbose=kwargs['verbose'])

    def load_chunk(self, chunk=None):
        self.x_lstg, self.lookup = chunk

    def generate_recorder(self):
        return DiscrimRecorder(verbose=self.verbose,
                               records_path=self.records_path,
                               record_sim=True)

    def generate_composer(self):
        return self._composer

    def generate_agent(self):
        model = self.ModelCls(**self.model_kwargs)
        model_path = self.run_dir + 'itr/{}/agent.net'.format(self.itr)
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
        relist_count = 0
        while True:
            environment.reset()
            sale, price, dur = environment.run()
            if sale:
                return slr_reward(price=price,
                                  start_price=environment.lookup[START_PRICE],
                                  meta=environment.lookup[META],
                                  elapsed=dur,
                                  relist_count=relist_count,
                                  # TODO: change to self.discount_rate
                                  discount_rate=MONTHLY_DISCOUNT)
            else:
                relist_count += 1

    @property
    def records_path(self):
        return self.run_dir + 'outcomes/{}.gz'.format(self.num)
