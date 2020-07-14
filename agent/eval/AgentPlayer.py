from rlenv.util import sample_categorical
from rlenv.interfaces.PlayerInterface import PlayerInterface, \
    SimulatedSeller, SimulatedBuyer


class AgentPlayer(PlayerInterface):
    def __init__(self, agent_model=None):
        super().__init__(byr=agent_model.byr, agent=True)
        self.agent_model = agent_model
        if not self.byr:
            self.delay_simulator = SimulatedSeller(full=False)
        else:
            self.delay_simulator = SimulatedBuyer(full=False)

    def sample_con(self, params=None, turn=None):
        return sample_categorical(params)

    def sample_msg(self, params=None, turn=None):
        return 0.0

    def sample_delay(self, params=None, turn=None):
        return self.delay_simulator.sample_delay(params=params)

    def query_con(self, input_dict=None, turn=None):
        return self.agent_model.con(input_dict=input_dict)

    def query_msg(self, input_dict=None, turn=None):
        return None

    def query_delay(self, input_dict=None, turn=None):
        return self.delay_simulator.query_delay(input_dict=input_dict,
                                                turn=turn)
