import pandas as pd
from featnames import LSTG


class LstgLoader:
    """
    Abstract class to pass lstg data from an arbitrary data source
    to the environment and generator as necessary
    """
    def __init__(self):
        self.lookup = None
        self.p_arrival = None
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
    def __init__(self, x_lstg=None, lookup=None, p_arrival=None):
        """
        :param pd.DataFrame x_lstg:
        :param pd.DataFrame lookup:
        """
        super().__init__()
        self._x_lstg_slice = x_lstg
        self._lookup_slice = lookup.reset_index(drop=False)
        self._p_arrival_slice = p_arrival
        self._ix = 0
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
            self.x_lstg = self._x_lstg_slice.iloc[self._ix, :]
            self.p_arrival = self._p_arrival_slice.iloc[self._ix, :]
            self.lstg = int(self.lookup[LSTG])
            self._ix += 1
            return self.x_lstg, self.lookup, self.p_arrival
        else:
            raise RuntimeError("Exhausted lstgs")

    def has_next(self):
        self.verify_init()
        return self._ix < self._num_lstgs

    def init(self, rank):
        pass

    def did_init(self):
        return True

    def __len__(self):
        return self._num_lstgs

    @property
    def x_lstg_cols(self):
        return list(self._x_lstg_slice.columns)
