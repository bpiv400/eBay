from rlenv.events.Thread import Thread


class SellerThread(Thread):
    def __init__(self, priority=None, thread_id=None, buyer=None):
        super(SellerThread, self).__init__(priority=priority, thread_id=thread_id)
        self.buyer = buyer
