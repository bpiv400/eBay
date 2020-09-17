from constants import MAX_DELAY_TURN
from rlenv.EBayEnv import EBayEnv
from rlenv.util import get_con_outcomes


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


class NoSlrExpEnv(SimulatorEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reject_next = dict()  # use thread_id as key

    def process_delay(self, event):
        if event.turn % 2 == 0:
            # get a delay
            delay_kwargs = dict(input_dict=self.get_delay_input_dict(event),
                                turn=event.turn,
                                thread_id=event.thread_id,
                                time=event.priority)
            delay_seconds = self.get_delay(**delay_kwargs)

            # if the delay is an expiration, try again until it isn't
            self.reject_next[event.thread_id] = False
            while delay_seconds == MAX_DELAY_TURN:
                self.reject_next[event.thread_id] = True
                delay_seconds = self.get_delay(**delay_kwargs)

            # update the thread and put event in queue
            event.update_delay(seconds=delay_seconds)
            self.queue.push(event)
            return False
        else:
            return super().process_delay(event)

    def process_offer(self, event):
        if event.turn % 2 == 0:
            # check whether the lstg expired, censoring this offer
            if self.is_lstg_expired(event):
                return self.process_lstg_expiration(event)

            # otherwise check whether this offer corresponds to an expiration rej
            assert not event.thread_expired()

            # update thread features
            self.prepare_offer(event)

            # reject if previously sampled an expiration
            if self.reject_next[event.thread_id]:
                con_outcomes = get_con_outcomes(con=0,
                                                sources=event.sources(),
                                                turn=event.turn)
                # update features
                offer = event.update_con_outcomes(con_outcomes=con_outcomes)

            # otherwise generate a standard offer
            else:
                offer = self.get_offer_outcomes(event, slr=True)

            self.reject_next[event.thread_id] = None  # reset to bool in process_delay()

            return self.process_post_offer(event, offer)
        else:
            return super().process_offer(event)
