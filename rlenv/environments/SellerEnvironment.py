from collections import namedtuple
import pandas as pd
import numpy as np
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlenv.environments.EbayEnvironment import EbayEnvironment
from rlenv.events import event_types
from rlenv.events.SellerThread import SellerThread
from rlenv.env_consts import (SELLER_HORIZON, LOOKUP, X_LSTG, 
                              ACTION_SPACE_NAME, OBS_SPACE_NAME,
                              THREAD_MAP, LSTG_MAP, TURN_IND_MAP)
from rlenv.simulators import SimulatedBuyer
from rlenv.spaces.ConSpace import ConSpace
from featnames import START_TIME, CON, DELAY
from constants import MONTH


class SellerEnvironment(EbayEnvironment, Env):
    def __init__(self, params):
        super(SellerEnvironment, self).__init__(
            arrival=params['arrival'], verbose=params['verbose'])
        # attributes for getting lstg data
        self._file = params['file']
        self._num_lstgs = len(self._file[LOOKUP])
        self._lookup_cols = self._file[LOOKUP].attrs['cols']
        self._lookup_cols = [col.decode('utf-8') for col in self._lookup_cols]
        self._lookup_slice, self._x_lstg_slice = None, None
        self._ix = -1
        # action and observation spaces
        self.agent_delay = params['delay']
        self._action_space = self._define_action_space()
        self._observation_space = self._define_observation_space()
        # buyer model
        self.buyer = params['buyer']

        # features for interacting with the agent
        self._last_event = None

    def reset(self):
        self._reset_lstg()
        super(SellerEnvironment, self).reset()
        lstg_complete = super(SellerEnvironment, self).run()
        if lstg_complete:
            return self.reset()
        else:
            return self._get_obs()

    def _check_complete(self, event):
        if event.type == event_types.SELLER_DELAY:
            return True
        else:
            return False

    def run(self):
        lstg_complete = super(SellerEnvironment, self).run()
        return self._get_obs(), self._get_reward(), lstg_complete, self._get_info()

    def step(self, action):
        # update
        self._last_event.seller_offer(action)
        self.queue.push(self._last_event)
        self._last_event = None
        return self.run()

    def _process_slr_delay(self, event):
        if self._lstg_expiration(event):
            return True
        else:
            # need to store remaining
            self._last_event = event
            event.init_delay()

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
        self.end_time = self.lookup[START_TIME] + MONTH

    def _draw_lstgs(self):
        ids = np.random.randint(self._num_lstgs)
        self._lookup_slice = self._file[LOOKUP][ids, :]
        self._x_lstg_slice = self._file[X_LSTG][ids, :]
        self._ix = 0

    def _record(self, event, start_thread=None, byr_hist=None):
        pass

    @property
    def horizon(self):
        return SELLER_HORIZON

    def _define_action_space(self):
        # message not included because agent can't write a msg
        con = ConSpace()
        if self.agent_delay:
            nt = namedtuple(ACTION_SPACE_NAME, [CON, DELAY])
            delay = FloatBox([0.0], [1.0], null_value=0)
            return Composite([con, delay], nt)
        else:
            nt = namedtuple(ACTION_SPACE_NAME, [CON])
            return Composite([con], nt)

    def _define_observation_space(self):
        feat_counts = self.arrival.composer.feat_counts
        nt = namedtuple(OBS_SPACE_NAME, [LSTG_MAP, THREAD_MAP, TURN_IND_MAP, ])
        lstg = FloatBox(0, 100, shape=(len(feat_counts[LSTG_MAP]),))
        thread = FloatBox(0, 100, shape=(len(feat_counts[THREAD_MAP]),))
        turn = FloatBox(0, 100, shape=(len(feat_counts[TURN_IND_MAP]),))
        remain = FloatBox(0, 1, shape=(1, ))
        return Composite([lstg, thread, turn, remain], OBS_SPACE)

    def make_thread(self, priority):
        return SellerThread(priority=priority, thread_id=self.thread_counter,
                            buyer=SimulatedBuyer(model=self.buyer))

    def _get_obs(self):
        return self._last_event.get_obs()

    def _get_reward(self):
        raise NotImplementedError("After Etan discussion")

    def _get_info(self):
        return None





