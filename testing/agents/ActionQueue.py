from rlenv.Heap import Heap
from testing.agents.Action import Action
from testing.util import compare_input_dicts
from utils import get_role


class ActionQueue:
    def __init__(self, byr=False):
        self.byr = byr
        self.action_queue = Heap(entry_type=Action)
        self.last_action = None

    def push_action(self, action=None):
        self.action_queue.push(action)

    def get_buyer_arrival(self):
        return self.action_queue.peek().time

    def get_agent_con(self, agent_tuple=None):
        obs, _, _, info = agent_tuple
        obs = obs._asdict()

        # check that turn matches agents type
        next_action = self.action_queue.pop()  # type: Action

        # if the tuple isn't a dummy generated by reset wrapper
        if info is not None:
            # check that turns are correct
            assert info.turn % 2 == int(self.byr)
            assert self.last_action.turn == info.turn

            # check that thread id matches
            if self.last_action.thread_id != info.thread_id:
                raise RuntimeError('Invalid thread ids: ({}, {})'.format(
                    self.last_action.thread_id, info.thread_id))

            # check that time is the same
            if self.last_action.time != info.priority:
                raise RuntimeError('Different priorities: ({}, {})'.format(
                    self.last_action.time, info.priority))

        elif self.byr:
            assert next_action.turn == 1

        # check agent inputs
        if not next_action.censored:
            for feat_set in obs.keys():
                obs[feat_set] = obs[feat_set].unsqueeze(0)
            compare_input_dicts(model=get_role(self.byr),
                                stored_inputs=next_action.input_dict,
                                env_inputs=obs)

        # save for next info tuple
        self.last_action = next_action

        return next_action.con

    def verify_done(self):
        while not self.action_queue.empty:
            next_action = self.action_queue.pop()
            if not next_action.censored:
                raise RuntimeError('Thread {}, turn {} not censored'.format(
                    next_action.thread_id, next_action.turn))
