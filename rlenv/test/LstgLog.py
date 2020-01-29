from rlenv.test.ThreadLog import ThreadLog

class LstgLog:

    def __init__(self, params=None):
        """
        # TODO: Update
        :param params: A format decided upon by Etan and I in meeting
        """
        self.params = params
        self.arrivals = self.generate_arrival_logs()
        self.threads = dict()
        self.x_lstg = None # TODO: Create a dictionary containing all the x_lstg components that are common
        # among all models / turns (e.g. all but x_lstg)
        for arrival in self.arrivals.keys():
            self.threads[arrival] = self.generate_thread_log(thread=arrival)

    def generate_arrival_logs(self):
        print(self.params)
        arrival_logs = dict()
        # TODO: Create an ArrivalLog for each arrival containing the necessary outcome data
        # TODO: and Arrival Model / Hist Model inputs
        return arrival_logs

    def generate_thread_log(self, thread=None):
        print(self.params)
        # TODO: Creates a ThreadLog for the given thread containing outcome data for each turn
        # TODO: of the the thread & model input data for instance a model is run
        # subset params
        return dict()

    def get_con(self, event=None):
        """

        :param rlenv.Events.Thread.Thread event:
        :return: np.float
        """
        con = self.threads[event.thread_id].get_con(event=event, x_lstg=self.x_lstg)


