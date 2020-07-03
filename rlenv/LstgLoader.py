import os
import h5py
import numpy as np
import pandas as pd
from agent.util import get_train_file_path
from featnames import LSTG
from rlenv.const import LOOKUP, X_LSTG, ENV_LSTG_COUNT


class LstgLoader:
    """
    Abstract class to pass lstg data from an arbitrary data source
    to the environment and generator as necessary
    """
    def __init__(self):
        self.lookup = None
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

    def init(self):
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
    def __init__(self, x_lstg=None, lookup=None):
        """
        :param pd.DataFrame x_lstg:
        :param pd.DataFrame lookup:
        """
        super().__init__()
        self._x_lstg_slice = x_lstg
        self._lookup_slice = lookup.reset_index(drop=False)
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
            self.lstg = self.lookup[LSTG]
            self._ix += 1
            return self.x_lstg, self.lookup
        else:
            raise RuntimeError("Exhausted lstgs")

    def has_next(self):
        self.verify_init()
        return self._ix < self._num_lstgs

    def init(self):
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
        if 'filename' not in kwargs:
            self._filename = get_train_file_path(0)
        else:
            self._filename = kwargs['filename']
        self._file = None
        self._is_init = False
        self._num_lstgs = None
        self._lookup_cols = None
        self._x_lstg_cols = kwargs['x_lstg_cols']
        self._lookup_slice, self._x_lstg_slice = None, None
        self._ix = -1
        self._file_opened = False

    def next_lstg(self):
        self.verify_init()
        if self._cache_empty():
            self._draw_lstgs()

        x_lstg = pd.Series(self._x_lstg_slice[self._ix, :],
                           index=self._x_lstg_cols)
        lookup = pd.Series(self._lookup_slice[self._ix, :],
                           index=self._lookup_cols)
        self.x_lstg = x_lstg
        self.lookup = lookup
        self.lstg = self.lookup.loc[self._ix, LSTG]
        self._ix += 1
        return self.x_lstg, self.lookup

    def _cache_empty(self):
        return self._ix == -1 or\
               self._ix == self._lookup_slice.shape[0]

    def has_next(self):
        self.verify_init()
        return True

    @property
    def did_init(self):
        return self._file_opened

    def next_id(self):
        self.verify_init()
        if self._cache_empty():
            self._draw_lstgs()
        lstg_col = self._lookup_cols.index(LSTG)
        return int(self._lookup_slice[self._ix, lstg_col])

    def init(self):
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        self._file = h5py.File(self._filename, "r")
        self._num_lstgs = len(self._file[LOOKUP])
        self._lookup_cols = self._file[LOOKUP].attrs['cols']
        self._lookup_cols = [col.decode('utf-8') for col in self._lookup_cols]
        self._file_opened = True

    def _draw_lstgs(self):
        ids = np.random.choice(self._num_lstgs, ENV_LSTG_COUNT,
                               replace=False)
        reordering = np.argsort(ids)
        sorted_ids = ids[reordering]
        unsorted_ids = np.argsort(reordering)
        self._lookup_slice = self._file[LOOKUP][sorted_ids, :]
        self._x_lstg_slice = self._file[X_LSTG][sorted_ids, :]
        self._lookup_slice = self._lookup_slice[unsorted_ids, :]
        self._x_lstg_slice = self._x_lstg_slice[unsorted_ids, :]
        self._ix = 0
