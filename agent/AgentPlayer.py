from rlenv.interfaces.PlayerInterface import PlayerInterface


class AgentPlayer(PlayerInterface):
    def __init__(self, param_path=None):
        self.params = torch.load(param_path)[]
    def sample_con(self, params=None):
        pass

    def sample_msg(self, params=None):
        pass

    def sample_delay(self, params=None):
        pass

    def query_con(self, input_dict=None, turn=None):
        pass

    def query_msg(self, input_dict=None, turn=None):
        pass

    def query_delay(self, input_dict=None, turn=None):
        pass