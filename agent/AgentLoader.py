import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from constants import NUM_CHUNKS
from featnames import TRAIN_RL, X_LSTG, LOOKUP, ARRIVALS
from env.LstgLoader import LstgLoader, ChunkLoader
from utils import load_chunk


class AgentLoader(LstgLoader):
    def __init__(self):
        super().__init__()

        # to be initialized later
        self._x_lstg_slice = self._lookup_slice = self._arrivals_slice = None
        self._internal_loader = None

    def init(self, rank):
        self._load_chunks(rank)
        self._draw_lstgs()

    def _load_chunks(self, rank=None):
        nums = np.split(np.arange(NUM_CHUNKS), mp.cpu_count())[rank]
        x_lstg, lookup, arrivals = None, [], {}
        for i, num in enumerate(nums):
            chunk = load_chunk(part=TRAIN_RL, num=num)
            if i == 0:
                x_lstg = chunk[X_LSTG]
            else:
                for k, v in x_lstg.items():
                    x_lstg[k] = pd.concat([v, chunk[X_LSTG][k]])
            lookup.append(chunk[LOOKUP])
            arrivals.update(chunk[ARRIVALS])
        self._x_lstg_slice = x_lstg
        self._lookup_slice = pd.concat(lookup, axis=0)
        self._arrivals_slice = arrivals

    def next_lstg(self):
        self.verify_init()
        if self._cache_empty():
            self._draw_lstgs()
        x_lstg, lookup, arrivals = self._internal_loader.next_lstg()
        self.lstg = self._internal_loader.lstg
        return x_lstg, lookup, arrivals

    def _cache_empty(self):
        return not self._internal_loader.has_next()

    def has_next(self):
        if not self.did_init:
            self.init(0)
        return True

    @property
    def did_init(self):
        return self._lookup_slice is not None

    @property
    def sim(self):
        return self._internal_loader.sim

    def next_id(self):
        self.verify_init()
        if self._cache_empty():
            self._draw_lstgs()
        return self._internal_loader.next_id()

    def _draw_lstgs(self):
        lstgs = np.array(self._lookup_slice.index)
        np.random.shuffle(lstgs)
        self._x_lstg_slice = {k: v.reindex(lstgs)
                              for k, v in self._x_lstg_slice.items()}
        self._lookup_slice = self._lookup_slice.reindex(lstgs)
        self._internal_loader = ChunkLoader(
            x_lstg=self._x_lstg_slice,
            lookup=self._lookup_slice,
            arrivals=self._arrivals_slice
        )
