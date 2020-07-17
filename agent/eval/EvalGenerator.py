from rlenv.generate.Generator import Generator
from rlenv.generate.Recorder import OutcomeRecorder
from agent.eval.AgentPlayer import AgentPlayer
from agent.util import load_agent_model
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from agent.AgentComposer import AgentComposer
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller


class EvalGenerator(Generator):
    def __init__(self, **kwargs):
        super().__init__(verbose=kwargs['verbose'])
        self.agent_params = kwargs['agent_params']
        self.model_kwargs = kwargs['model_kwargs']
        self.run_dir = kwargs['run_dir']

    def generate_recorder(self):
        return OutcomeRecorder(verbose=self.verbose,
                               record_sim=True)

    def generate_composer(self):
        return AgentComposer(agent_params=self.agent_params)

    def generate_agent(self):
        model = PgCategoricalAgentModel(**self.model_kwargs)
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

    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        for i, lstg in enumerate(self.x_lstg.index):
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            environment = self.setup_env(lstg=lstg, lookup=lookup)

            # update listing in recorder
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)

            # simulate lstg until sale
            self.simulate_lstg(environment)

        # save the recorder
        self.recorder.dump()

    def simulate_lstg(self, env):
        """
        Simulates a particular listing either once or until sale.
        :param env: SimulatorEnvironment
        :return: outcome tuple
        """
        while True:
            env.reset()
            outcome = env.run()
            if outcome.sale:
                return outcome

    @property
    def records_path(self):
        return self.run_dir + '{}/{}.gz'.format(self.part, self.chunk)
