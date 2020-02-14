import numpy as np
from rlpyt.spaces.base import Space
from agent.agent_utils import get_con_set


class ConSpace(Space):
    def __init__(self, con_type=None):
        self.con_type = con_type
        self.con_set = get_con_set(con_type)
        self.dtype = np.float32
        self._null_value = 0.0
        self.shape = (1, )

    def sample(self):
        index = np.random.randint(0, self.con_set.size, 1)[0]
        return self.con_set[index]

    def null_value(self):
        return self._null_value

    @property
    def bounds(self):
        return 0.0, 1.0

    @property
    def n(self):
        """Number of elements in the space"""
        return self.con_set.size

    def __repr__(self):
        return "ConSpace({})".format(self.con_set)
