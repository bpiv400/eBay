import numpy as np
import pandas as pd
from constants import RL_BYR, RL_SLR, PARTS_DIR, NUM_CHUNKS
from featnames import BYR, LSTG
from rlenv.util import load_chunk


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


class TrainLoader(LstgLoader):
    def __init__(self, **kwargs):
        super().__init__()
        self.part = RL_BYR if kwargs[BYR] else RL_SLR

        # to be initialized later
        self._x_lstg_slice = self._lookup_slice = self._p_arrival_slice = None
        self._internal_loader = None

    def init(self, rank):
        filename = self._get_train_file_path(rank)
        chunk = load_chunk(input_path=filename)
        self._x_lstg_slice, self._lookup_slice, self._p_arrival_slice = chunk
        self._draw_lstgs()

    def _get_train_file_path(self, rank=None):
        rank = rank % NUM_CHUNKS  # for using more workers
        return PARTS_DIR + '{}/chunks/{}.pkl'.format(self.part, rank)

    def next_lstg(self):
        self.verify_init()
        if self._cache_empty():
            self._draw_lstgs()
        x_lstg, lookup, p_arrival = self._internal_loader.next_lstg()
        self.lstg = self._internal_loader.lstg
        return x_lstg, lookup, p_arrival

    def _cache_empty(self):
        return not self._internal_loader.has_next()

    def has_next(self):
        if not self.did_init:
            self.init(0)
        return True

    @property
    def x_lstg_cols(self):
        return self._internal_loader.x_lstg_cols

    @property
    def did_init(self):
        return self._lookup_slice is not None

    def next_id(self):
        self.verify_init()
        if self._cache_empty():
            self._draw_lstgs()
        return self._internal_loader.next_id()

    def _draw_lstgs(self):
        lstgs = np.array(self._lookup_slice.index)
        np.random.shuffle(lstgs)
        self._x_lstg_slice = self._x_lstg_slice.reindex(lstgs)
        self._p_arrival_slice = self._p_arrival_slice.reindex(lstgs)
        self._lookup_slice = self._lookup_slice.reindex(lstgs)
        self._internal_loader = ChunkLoader(
            x_lstg=self._x_lstg_slice,
            lookup=self._lookup_slice,
            p_arrival=self._p_arrival_slice
        )
