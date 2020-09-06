from rlpyt.utils.collections import namedarraytuple
from rlenv.const import DELAY_EVENT, OFFER_EVENT
from agent.envs.AgentEnv import AgentEnv, EventLog
from agent.util import define_con_set
from rlenv.util import get_delay_outcomes, get_con_outcomes
from utils import load_sizes
from rlenv.const import DELAY_IND
from constants import POLICY_SLR, MAX_DELAY_TURN
from featnames import START_PRICE

SellerObs = namedarraytuple("SellerObs",
                            list(load_sizes(POLICY_SLR)['x'].keys()))


class SellerEnv(AgentEnv):

    def is_agent_turn(self, event):
        """
        Checks whether the agents should take a turn
        :param rlenv.events.Thread.Thread event:
        :return: bool
        """
        if event.turn % 2 == 0:
            if event.type == DELAY_EVENT:
                return True
            if event.type == OFFER_EVENT:
                return not(self.is_lstg_expired(event) or event.thread_expired())
        return False

    def reset(self, next_lstg=True):
        self.init_reset(next_lstg=next_lstg)  # in AgentEnvironment
        while True:
            event, lstg_complete = super().run()  # calls EBayEnvironment.run()

            # time to sample an agents action
            if not lstg_complete:
                if event.type == DELAY_EVENT:  # draw delay
                    self._process_slr_delay(event)
                else:
                    self.curr_event = event
                    self.prepare_offer(event)
                    return self.get_obs(event=event, done=False)

            # if the lstg is complete
            elif next_lstg:  # queue up next lstg in training
                self.init_reset(next_lstg=True)
            else:
                return None  # for EvalGenerator

    def init_reset(self, next_lstg=None):
        super().init_reset(next_lstg=next_lstg)
        self.item_value = self.lookup[START_PRICE]

    def step(self, action):
        """
        Process int giving concession/delay
        :param action: int returned from agents
        :return: tuple described in rlenv
        """
        self.last_event = EventLog(priority=self.curr_event.priority,
                                   thread_id=self.curr_event.thread_id,
                                   turn=self.curr_event.turn)
        con = self.turn_from_action(action)
        if self.verbose:
            print('AGENT TURN: con: {}'.format(con))

        # execute offer
        self.num_offers += 1
        if con <= 1:
            con_outcomes = get_con_outcomes(con=con,
                                            sources=self.curr_event.sources(),
                                            turn=self.curr_event.turn)
            offer = self.curr_event.update_con_outcomes(con_outcomes)
            lstg_complete = self.process_post_offer(self.curr_event, offer)
            if lstg_complete:
                return self.agent_tuple(event=self.curr_event,
                                        done=lstg_complete,
                                        info=self.get_info(self.last_event))
        else:
            # get initial delay from sources
            turn = self.curr_event.turn
            key = 'offer{}'.format(turn)
            last_delay = self.curr_event.sources()[key][DELAY_IND]
            assert 0 < last_delay < 1
            last_delay_seconds = int(round(last_delay * MAX_DELAY_TURN))

            # update sources with expiration delay
            delay_outcomes = get_delay_outcomes(seconds=MAX_DELAY_TURN,
                                                turn=turn)
            self.curr_event.sources.update_delay(delay_outcomes=delay_outcomes,
                                                 turn=turn)

            # update priority and push thread to queue
            self.curr_event.priority += MAX_DELAY_TURN - last_delay_seconds
            self.queue.push(self.curr_event)

        return self.run()

    def run(self):  # until EbayEnvironment.run() stops at agents turn
        while True:
            event, lstg_complete = super().run()
            if event.type == DELAY_EVENT:
                self._process_slr_delay(event)
            else:
                self.curr_event = event  # save for step method
                if not lstg_complete:
                    self.prepare_offer(event)
                return self.agent_tuple(done=lstg_complete,
                                        event=event,
                                        info=self.get_info(self.last_event))

    def _process_slr_delay(self, event):
        delay_seconds = self.draw_agent_delay(event)
        if self.verbose:
            print('AGENT TURN: delay (sec) : {}'.format(delay_seconds))
        event.update_delay(seconds=delay_seconds)
        self.queue.push(event)

    def get_reward(self):
        if self.outcome is None or not self.outcome.sale:
            return 0.
        return self.outcome.price

    def _define_con_set(self, con_set):
        return define_con_set(con_set=con_set, byr=False)

    @property
    def horizon(self):
        return 100

    @property
    def _obs_class(self):
        return SellerObs
