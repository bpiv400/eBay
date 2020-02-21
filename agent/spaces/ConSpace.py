import numpy as np
from rlpyt.spaces.base import Space


class ConSpace(Space):
    def __init__(self, con_set=None):
        self.con_set = con_set
        self.dtype = np.int64
        self._null_value = 0
        self.shape = ()

    def sample(self):
        index = np.random.randint(0, self.con_set.size)
        return np.array(index)

    def null_value(self):
        return np.array(self._null_value, dtype=np.int64)

    @property
    def bounds(self):
        return 0, self.con_set.size

    @property
    def n(self):
        """Number of elements in the space"""
        return self.con_set.size

    def __repr__(self):
        return "ConSpace({})".format(self.con_set)
