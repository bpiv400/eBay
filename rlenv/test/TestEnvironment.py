from rlenv.environments.EbayEnvironment import EbayEnvironment


class TestEnvironment(EbayEnvironment):
    def __init__(self, params=None):
        super().__init__(params=params)
        self.lstg_log = params['log']

    def _record(self, event, byr_hist=None, censored=None):
        pass
