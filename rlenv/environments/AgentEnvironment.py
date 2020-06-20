from collections import namedtuple
from featnames import START_PRICE
from constants import MONTH
from utils import get_months_since_lstg
from agent.spaces.ConSpace import ConSpace
from inputs.const import INTERVAL_CT_TURN, INTERVAL_TURN
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlenv.environments.EbayEnvironment import EbayEnvironment
from rlenv.Recorder import Recorder
from agent.util import get_con_set

InfoTraj = namedtuple("InfoTraj", ["months", "bin_proceeds", "done"])


class AgentEnvironment(EbayEnvironment, Env):

    def __init__(self, **kwargs):
        super().__init__(params=kwargs)
        self.last_event = None  # type: Thread
        # action and observation spaces
        self.con_set = get_con_set(con=self.composer.con_type,
                                   byr=self.composer.byr,
                                   delay=self.composer.delay)
        self._action_space = self.define_action_space()
        self._obs_class = self.define_observation_class()
        self._observation_space = self.define_observation_space()

    @property
    def delay(self):
        return self.composer.delay

    def define_observation_class(self):
        raise NotImplementedError("Please extend")

    def define_observation_space(self):
        sizes = self.composer.agent_sizes['x']
        boxes = [FloatBox(-1000, 1000, shape=size) for size in sizes.values()]
        return Composite(boxes, self._obs_class)

    def agent_tuple(self, done=None):
        obs = self.get_obs(sources=self.last_event.sources(),
                           turn=self.last_event.turn)
        months = (self.last_event.priority - self.start_time) / MONTH
        months += self.relist_count  # add in months without sale
        bin_proceeds = (1-self.cut) * self.lookup[START_PRICE]
        info = InfoTraj(months=months, bin_proceeds=bin_proceeds,
                        done=done)
        reward = self.get_reward()
        return obs, reward, done, info

    def get_obs(self, sources=None, turn=None):
        if sources is None or turn is None:
            raise RuntimeError("Missing arguments to get observation")
        obs_dict = self.composer.build_input_dict(model_name=None,
                                                  sources=sources,
                                                  turn=turn)
        return self._obs_class(**obs_dict)

    def get_offer_time(self, event):
        # query with delay model
        input_dict = self.get_delay_input_dict(event=event)
        intervals = (self.end_time - event.priority) / INTERVAL_TURN
        max_interval = min(int(intervals), INTERVAL_CT_TURN)
        delay = self.get_delay(input_dict=input_dict, turn=event.turn,
                               thread_id=event.thread_id, time=event.priority,
                               max_interval=max(1, max_interval))
        return max(delay, 1) + event.priority

    def process_rl_offer(self, event):
        """
        :param RlThread event:
        :return: bool indicating the lstg is over
        """
        # check whether the lstg expired, censoring this offer
        if self.is_lstg_expired(event):
            return self.process_lstg_expiration(event)
        slr_offer = event.turn % 2 == 0
        if event.thread_expired():
            if slr_offer:
                self.process_slr_expire(event)
                return False
            else:
                raise RuntimeError("Thread should never expire before"
                                   "buyer agent offer")
        time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                               time=event.priority)
        months_since_lstg = None
        if event.turn == 1:
            months_since_lstg = get_months_since_lstg(lstg_start=self.start_time,
                                                      time=event.priority)
        event.init_rl_offer(months_since_lstg=months_since_lstg, time_feats=time_feats)
        offer = event.execute_offer()
        return self.process_post_offer(event, offer)

    def turn_from_action(self, action=None):
        return self.con_set[action]

    def get_reward(self):
        raise NotImplementedError()

    @property
    def horizon(self):
        pass

    def define_action_space(self):
        return ConSpace(con_set=self.con_set)

    # TODO: may update
    def record(self, event, byr_hist=None, censored=False):
        if not censored and byr_hist is None:
            if self.verbose:
                Recorder.print_offer(event)

    def is_agent_turn(self, event):
        raise NotImplementedError()

    def step(self, action):
        """
        Process float giving concession
        :param action: float returned from agent
        :return:
        """
        raise NotImplementedError()



