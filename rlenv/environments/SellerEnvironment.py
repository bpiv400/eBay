import pandas as pd
import numpy as np
from rlpyt.envs.base import Env
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlenv.environments.EbayEnvironment import EbayEnvironment
from rlenv.composer.maps import THREAD_MAP, LSTG_MAP, TURN_IND_MAP
from rlenv.events import event_types
from rlenv.env_consts import SELLER_HORIZON, LOOKUP, X_LSTG, START_DAY, MONTH
from rlenv.spaces.ConSpace import ConSpace
from collections import namedtuple


class SellerEnvironment(EbayEnvironment, Env):
    def __init__(self, arrival, file):
        super(SellerEnvironment, self).__init__(arrival)
        # attributes for getting lstg data
        self._file = file
        self._num_lstgs = len(self._file[LOOKUP])
        self._lookup_cols = self._file[LOOKUP].attrs['cols']
        self._lookup_cols = [col.decode('utf-8') for col in self._lookup_cols]
        self._lookup_slice, self._x_lstg_slice = None, None
        self._ix = -1
        # action and observation spaces
        self._action_space = self._define_action_space()
        self._observation_space = self._define_observation_space()
        # recent event
        self.recent_event = None

    def reset(self):
        self._reset_lstg()
        self.end_time = self.lookup[START_DAY] + MONTH
        super(SellerEnvironment, self).reset()
        lstg_complete = super(SellerEnvironment, self).run()
        if lstg_complete:
            return self.reset()
        else:
            return self._prepare_obs()

    def _check_complete(self, event):
        if event.type == event_types.SELLER_DELAY:
            return True
        else:
            return False

    def run(self):
        lstg_complete = super(SellerEnvironment, self).run()
        done = lstg_complete
        return self._prepare_obs(), self._reward(), done, self._info()

    def step(self, action):
        pass

    def _reset_lstg(self):
        """
        Sample a new lstg from the file and set lookup and x_lstg series
        """
        if self._ix == -1 or self._ix == self._num_lstgs:
            self._draw_lstgs()
        self.x_lstg = pd.Series(self._x_lstg_slice[self._ix, :],
                                index=self.arrival.composer.x_lstg)
        self.lookup = pd.Series(self._lookup_slice[self._ix, :], index=self._lookup_cols)
        self._ix += 1

    def _draw_lstgs(self):
        ids = np.random.randint(self._num_lstgs)
        self._lookup_slice = self._file[LOOKUP][ids, :]
        self._x_lstg_slice = self._file[X_LSTG][ids, :]
        self._ix = 0

    def _prepare_obs(self):
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

    def make_thread(self, priority):
        raise NotImplementedError()

    def _reward(self):
        raise NotImplementedError()

    def _prepare_obs(self):
        raise NotImplementedError()

    def _info(self):
        raise NotImplementedError()





