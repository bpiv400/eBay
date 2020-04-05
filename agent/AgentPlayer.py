import torch
from torch.distributions.categorical import Categorical
from rlenv.env_utils import sample_categorical
from rlenv.interfaces.PlayerInterface import (PlayerInterface, SimulatedSeller,
                                              SimulatedBuyer)


class AgentPlayer(PlayerInterface):
    def __init__(self, agent_model=None):
        """

        :param agent.models.AgentModel.AgentModel agent_model:
        """
        super().__init__(byr=agent_model.byr)
        self.agent_model = agent_model
        if not self.agent_model.delay:
            if not self.byr:
                self.delay_simulator = SimulatedSeller(full=False)
            else:
                self.delay_simulator = SimulatedBuyer(full=False)

    def sample_con(self, params=None):
        """

        :param torch.FloatTensor params:
        :return:
        """
        shape = params.shape[0]
        con_class = sample_categorical(params)
        if shape == 3:
            con = con_class * 50
        elif shape == 5:
            con = con_class * 25
        else:
            con = con_class * 100
        return con / 100

    def sample_msg(self, params=None):
        return 0.0

    def sample_delay(self, params=None):
        return self.delay_simulator.sample_delay(params=params)

    def query_con(self, input_dict=None, turn=None):
        return self.agent_model.con(input_dict=input_dict)

    def query_msg(self, input_dict=None, turn=None):
        return None

    def query_delay(self, input_dict=None, turn=None):
        return self.delay_simulator.query_delay(input_dict=input_dict)