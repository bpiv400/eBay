from rlpyt.envs.base import Env
from rlenv.environments.EbayEnvironment import EbayEnvironment


class SellerEnvironment(EbayEnvironment, Env):
    def __init__(self, arrival):
        super(SellerEnvironment, self).__init__(arrival)

    def reset(self):
        pass

    def step(self, action):
        pass

    @property
    def horizon(self):
        return 3
