from rlenv.events.Thread import Thread
from rlenv.env_consts import SELLER_DELAY, BUYER_DELAY


class SellerThread(Thread):
    def __init__(self, priority=None, thread_id=None, buyer=None, interval_attrs=None):
        super(SellerThread, self).__init__(priority=priority, thread_id=thread_id, interval_attrs=interval_attrs)
        self.buyer = buyer
        self.seller = None

    def init_delay(self, lstg_start):
        self._init_delay_sources(lstg_start)
        if self.turn % 2 == 0:
            self.type = SELLER_DELAY
        else:
            self.type = BUYER_DELAY
            self._init_delay_hidden()

    def seller_offer(self, *args):
        """
        Update sources with the result of a seller offer and return an offer dictionary

        :param args: tuple containing action
        :return:
        """
        if len(args) == 0:
            raise RuntimeError()
        action = args[0]
        norm = self.sources.agent_offer(action, self.turn)