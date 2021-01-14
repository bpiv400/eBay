import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from constants import NUM_CHUNKS, PARTS_DIR
from featnames import TRAIN_RL
from rlenv.LstgLoader import LstgLoader, ChunkLoader
from rlenv.util import load_chunk


class AgentLoader(LstgLoader):
    def __init__(self):
        super().__init__()

        # to be initialized later
        self._x_lstg_slice = self._lookup_slice = self._p_arrival_slice = None
        self._internal_loader = None

    def init(self, rank):
        self._load_chunks(rank)
        self._draw_lstgs()

    def _load_chunks(self, rank=None):
        nums = np.split(np.arange(NUM_CHUNKS), mp.cpu_count())[rank]
        x_lstg, lookup, p_arrival = [], [], []
        for num in nums:
            path = PARTS_DIR + '{}/chunks/{}.pkl'.format(TRAIN_RL, num)
            chunk = load_chunk(input_path=path)
            x_lstg.append(chunk[0])
            lookup.append(chunk[1])
            p_arrival.append(chunk[2])
        self._x_lstg_slice = pd.concat(x_lstg, axis=0)
        self._lookup_slice = pd.concat(lookup, axis=0)
        self._p_arrival_slice = pd.concat(p_arrival, axis=0)

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
