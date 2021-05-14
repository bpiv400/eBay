import pandas as pd
from constants import ARRIVAL_SIMS
from featnames import LSTG


class LstgLoader:
    """
    Abstract class to pass lstg data from an arbitrary data source
    to the environment and generator as necessary
    """
    def __init__(self):
        self.lookup = None
        self.arrivals = None
        self.x_lstg = None
        self.lstg = None

    def next_lstg(self):
        raise NotImplementedError()

    def has_next(self):
        """
        Returns boolean for whether there are
        any more lstgs in the loader
        """
        raise NotImplementedError()

    def next_id(self):
        """
        Returns next lstg id or throws an error if
        there are no more
        """
        raise NotImplementedError()

    def init(self, rank):
        """
        Performs loader initialization if any
        """
        raise NotImplementedError()

    def verify_init(self):
        if not self.did_init:
            raise RuntimeError("Must initialize loader before"
                               " performing this operation")

    @property
    def did_init(self):
        raise NotImplementedError()


class ChunkLoader(LstgLoader):
    """
    Loads lstgs from a chunk or chunk subset
    """
    def __init__(self, x_lstg=None, lookup=None, arrivals=None, num_sims=None):
        """
        :param pd.DataFrame x_lstg:
        :param pd.DataFrame lookup:
        :param dict arrivals:
        :param int num_sims:
        """
        super().__init__()
        assert num_sims <= ARRIVAL_SIMS
        self._x_lstg_slice = x_lstg
        self._lookup_slice = lookup.reset_index(drop=False)
        self._arrivals_slice = arrivals
        self._ix = 0
        self.sim = 0
        self._num_sims = num_sims
        self._num_lstgs = len(lookup.index)

    def next_id(self):
        self.verify_init()
        if self.has_next():
            return self._lookup_slice.iloc[self._ix, LSTG]
        else:
            raise RuntimeError("Exhausted lstgs")

    def next_lstg(self):
        self.verify_init()
        if self.has_next():
            self.lookup = self._lookup_slice.iloc[self._ix, :]
            self.lstg = int(self.lookup[LSTG])
            self.x_lstg = {k: v.iloc[self._ix, :].values
                           for k, v in self._x_lstg_slice.items()}
            self.arrivals = self._arrivals_slice[self.lstg][self.sim]

            self._ix += 1
            if self._ix == self._num_lstgs:
                self._ix = 0
                self.sim += 1
            return self.x_lstg, self.lookup, self.arrivals
        else:
            raise RuntimeError("Exhausted lstgs")

    def has_next(self):
        self.verify_init()
        return self.sim < self._num_sims

    def init(self, rank):
        pass

    def did_init(self):
        return True

    def __len__(self):
        return self._num_lstgs

