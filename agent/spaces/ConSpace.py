import numpy as np
from rlpyt.spaces.int_box import IntBox


class ConSpace(IntBox):
    def __init__(self):
        super(ConSpace, self).__init__(0, 101, shape=1, null_value=0)
        self.dtype = np.float32

    def sample(self):
        i = super(ConSpace, self).sample()
        i = i / 100
        return i

    def __repr__(self):
        return "ConSpace({0.0, 0.01, 0.02, ..., 1.0})"
