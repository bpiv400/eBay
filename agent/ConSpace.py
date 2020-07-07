import numpy as np
from rlpyt.spaces.base import Space


class ConSpace(Space):
    def __init__(self, size=None):
        self.size = size
        self.dtype = np.int64
        self._null_value = 0
        self.shape = ()

    def sample(self):
        index = np.random.randint(0, self.size)
        return np.array(index)

    def null_value(self):
        return np.array(self._null_value, dtype=np.int64)

    @property
    def bounds(self):
        return 0, self.size

    @property
    def n(self):
        """Number of elements in the space"""
        return self.size

    def __repr__(self):
        return "ConSpace"
