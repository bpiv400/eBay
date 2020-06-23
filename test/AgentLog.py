from Heap import Heap

class AgentLog:
    def __init__(self, byr=False, thread_id=None):
        self.byr = byr
        self.thread_id = thread_id
        self.interarrival_time = None
        self.action_queue = deque

    def push_action(self, action=None):
        self.action_queue.append(action)

    def verify_action(self, agent_tuple=None):



class ActionLog:
    def __init__(self, con=None, input_dict=None, time=None):
        self.con = con
        self.input_dict = input_dict
        self.check_time = time