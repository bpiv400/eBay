from utils import get_months_since_lstg
from agent.ConSpace import ConSpace
from constants import INTERVAL_TURN, INTERVAL_CT_TURN
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment


class EvalEnvironment(SimulatorEnvironment):

    def __init__(self, **kwargs):
        super().__init__(params=kwargs)

        self.relist_count = 0
        self.last_event = None  # type: Thread

    def get_offer_time(self, event):
        # query with delay model
        input_dict = self.get_delay_input_dict(event=event)
        intervals = (self.end_time - event.priority) / INTERVAL_TURN
        max_interval = min(int(intervals), INTERVAL_CT_TURN)
        delay = self.get_delay(input_dict=input_dict,
                               turn=event.turn,
                               thread_id=event.thread_id,
                               time=event.priority,
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

    def define_action_space(self):
        return ConSpace(size=len(self.con_set))

    def is_agent_turn(self, event):
        raise NotImplementedError()

    def step(self, action):
        """
        Process float giving concession
        :param action: float returned from agent
        :return:
        """
        raise NotImplementedError()



