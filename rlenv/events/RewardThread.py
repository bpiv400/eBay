from rlenv.events.Thread import Thread


class RewardThread(Thread):
    def __init__(self, priority=None, thread_id=None, buyer=None, seller=None):
        super(RewardThread, self).__init__(priority=priority, thread_id=thread_id)
        self.buyer = buyer
        self.seller = seller
