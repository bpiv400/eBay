from rlenv.environments.EBayEnv import EBayEnv


class SimulatorEnv(EBayEnv):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def run(self):
        """
        Runs a simulation of a single lstg until sale or expiration

        :return: a 3-tuple of (bool, float, int) giving whether the listing sells,
        the amount it sells for if it sells, and the amount of time it took to sell
        """
        super().run()
        return self.outcome

    def is_agent_turn(self, event):
        return False
