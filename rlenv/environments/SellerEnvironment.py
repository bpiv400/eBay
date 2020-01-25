from collections import namedtuple
import pandas as pd
import numpy as np
from rlenv.Composer import AgentComposer
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlenv.environments.EbayEnvironment import EbayEnvironment
from rlenv.env_consts import (SELLER_HORIZON, LOOKUP, X_LSTG, OFFER_EVENT,
                              ACTION_SPACE_NAME, OBS_SPACE_NAME,
                              LSTG_MAP, TURN_IND_MAP, ENV_LSTG_COUNT)
from rlenv.env_utils import get_con_outcomes
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
        self._action_space = self._define_action_space()
        self._observation_space = self._define_observation_space
        # model interfaces and composer
        self._composer = params['composer']  # type: AgentComposer
        self.seller = params['seller']
        self.buyer = params['buyer']
        self._last_event = None  # type: Thread

    def reset(self):
        while True:
            self._reset_lstg()
            super().reset()
            event, lstg_complete = super().run()
            if not lstg_complete:
                self._last_event = event
                return self._composer.get_obs(event)

    def _is_agent_turn(self, event):
        """
        Checks whether the agent should take a turn
        :param rlenv.events.Thread.Thread event:
        :return: bool
        """
        if event.type == OFFER_EVENT and event.turn % 2 == 0:
            if self._is_lstg_expired(event):
                return False
            elif event.thread_expired():
                return False
            else:
                return True
        else:
            return False

    def run(self):
        event, lstg_complete = super(SellerEnvironment, self).run()
        self._last_event = event
        return self._agent_tuple(lstg_complete)

    def step(self, action):
        """
        Process float giving concession
        :param action: float returned from agent
        :return:
        """
        offer_outcomes = get_con_outcomes(con=action, sources=self._last_event.sources(),
                                          turn=self._last_event.turn)
        offer = self._last_event.update_offer(offer_outcomes=offer_outcomes)
        lstg_complete = self._process_post_offer(self._last_event, offer)
        if lstg_complete:
            return self._agent_tuple(lstg_complete)
        self._last_event = None
        return self.run()

    def _reset_lstg(self):
        """
        Sample a new lstg from the file and set lookup and x_lstg series
        """
        if self._ix == -1 or self._ix == self._num_lstgs:
            self._draw_lstgs()
        self.x_lstg = pd.Series(self._x_lstg_slice[self._ix, :],
                                index=self._composer.x_lstg)
        self.lookup = pd.Series(self._lookup_slice[self._ix, :], index=self._lookup_cols)
        self._ix += 1
        self.end_time = self.lookup[START_TIME] + MONTH

    def _draw_lstgs(self):
        ids = np.random.randint(0, self._num_lstgs, ENV_LSTG_COUNT)
        self._lookup_slice = self._file[LOOKUP][ids, :]
        self._x_lstg_slice = self._file[X_LSTG][ids, :]
        self._ix = 0

    def _record(self, event, start_thread=None, byr_hist=None):
        raise NotImplementedError("Double check method signature")

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
        sizes = self._composer.agent_sizes
        boxes = [FloatBox(-1000, 1000, shape=len(size)) for size in sizes.values()]
        return Composite(boxes, self._composer.obs_space_class)

    def _agent_tuple(self, lstg_complete):
        obs = self._composer.get_obs(self._last_event.sources())
        return obs, self._get_reward(), lstg_complete, self._get_info()

    def _get_reward(self):
        raise NotImplementedError("After Etan discussion")

    def _get_info(self):
        raise NotImplementedError("")





