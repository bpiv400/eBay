from rlenv.events.Thread import Thread
from rlenv.env_consts import SELLER_DELAY, BUYER_DELAY


class RewardThread(Thread):
    def __init__(self, priority=None, thread_id=None, buyer=None, seller=None, intervals=None):
        super(RewardThread, self).__init__(priority=priority, thread_id=thread_id, intervals=intervals)
        self.buyer = buyer
        self.seller = seller
