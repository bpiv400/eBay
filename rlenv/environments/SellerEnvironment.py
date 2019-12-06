from rlpyt.envs.base import Env
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlenv.environments.EbayEnvironment import EbayEnvironment
from rlenv.composer.maps import THREAD_MAP, LSTG_MAP, TURN_IND_MAP
from rlenv.env_consts import SELLER_HORIZON
from rlenv.spaces.ConSpace import ConSpace
from collections import namedtuple


class SellerEnvironment(EbayEnvironment, Env):
    def __init__(self, arrival, file):
        super(SellerEnvironment, self).__init__(arrival)
        self._file = file
        self._action_space = self._define_action_space()
        self._observation_space = self._define_observation_space()

    def reset(self):
        self._reset_lstg()

    def step(self, action):
        pass

    def _reset_lstg(self):
        """
        Sample a new lstg from the file and set lookup and x_lstg series
        """
        pass

    @property
    def horizon(self):
        return SELLER_HORIZON

    @staticmethod
    def _define_action_space():
        nt = namedtuple('NegotiationActionSpace', ['con', 'delay', 'msg'])
        msg = IntBox(0, 2, shape=(1, ), null_value=0)
        delay = FloatBox([0.0], [1.0], null_value=0)
        con = ConSpace()
        return Composite([con, delay, msg], nt)

    def _define_observation_space(self):
        feat_counts = self.arrival.composer.feat_counts
        lstg = FloatBox(0, 100, shape=(len(feat_counts[LSTG_MAP]),))
        thread = FloatBox(0, 100, shape=(len(feat_counts[THREAD_MAP]),))
        turn = FloatBox(0, 100, shape=(len(feat_counts[TURN_IND_MAP]),))
        nt = namedtuple('NegotiationObsSpace', [LSTG_MAP, THREAD_MAP, TURN_IND_MAP])
        return Composite([lstg, thread, turn], nt)





