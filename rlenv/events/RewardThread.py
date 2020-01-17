from rlenv.events.Thread import Thread
from rlenv.env_consts import SELLER_DELAY, BUYER_DELAY


class RewardThread(Thread):
    def __init__(self, priority=None, thread_id=None, buyer=None, seller=None, interval_attrs=None):
        super(RewardThread, self).__init__(priority=priority, thread_id=thread_id, interval_attrs=interval_attrs)
        self.buyer = buyer
        self.seller = seller
