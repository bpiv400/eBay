from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment


class TestEnvironment(SimulatorEnvironment):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.lstg_log = kwargs['log']

    def record(self, event, byr_hist=None, censored=None):
        """Record nothing"""
        pass

    def get_con(self, event=None):
        con = self.lstg_log.get_con(event)
        return con
