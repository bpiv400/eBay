import math
from agent.util import get_agent_name
from Heap import Heap
from test.util import compare_input_dicts


class AgentLog:
    def __init__(self, byr=False, delay=False):
        self.byr = byr
        self.delay = delay
        self.model_name = get_agent_name(byr=self.byr, delay=self.delay,
                                         policy=True)
        self.interarrival_time = None
        self.action_queue = Heap(entry_type=ActionLog)

    def push_action(self, action=None):
        self.action_queue.push(action)

    def get_action(self, agent_tuple=None):
        obs, _, done, info = agent_tuple
        obs = obs._asdict()
        # check that turn matches agent type
        assert (info.turn % 2 == 1 and self.byr) or (not self.byr and info.turn % 2 == 0)
        next_action = self.action_queue.pop()  # type: ActionLog
        # check that turns match
        assert next_action.turn == info.turn
        # if not the first turn, check that thread id matches
        if next_action.turn != 1:
            assert next_action.thread_id == info.thread_id
        # check that time is the same
        assert math.isclose(next_action.months, info.months)
        compare_input_dicts(model=self.model_name, stored_inputs=next_action.input_dict,
                            env_inputs=obs)
        # check that done flags are the same
        assert done == info.done
        # check that the queue is empty if this is done
        if done:
            assert self.action_queue.empty
        return next_action.con


class ActionLog:
    def __init__(self, con=None, input_dict=None, months=None,
                 thread_id=None, turn=None):
        self.con = con
        self.input_dict = input_dict
        self.months = months
        self.thread_id = thread_id
        self.turn = turn

    def __lt__(self, other):
        if self.months == other.months:
            raise RuntimeError("Multiple agent actions should not happen at the same time")
        else:
            return self.months < other.months