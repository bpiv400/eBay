from rlenv.events.Thread import Thread


class SellerThread(Thread):
    def __init__(self, priority=None, thread_id=None, buyer=None, interval_attrs=None):
        super(SellerThread, self).__init__(priority=priority, thread_id=thread_id, interval_attrs=interval_attrs)
        self.buyer = buyer
        self.seller = None

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