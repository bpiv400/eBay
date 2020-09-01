from rlenv.LstgLoader import ChunkLoader
from testing.util import subset_inputs, subset_df
from featnames import X_LSTG, X_OFFER, X_THREAD, LOOKUP, P_ARRIVAL


class TestLoader(ChunkLoader):
    def __init__(self, chunk=None):
        super().__init__(x_lstg=chunk[X_LSTG],
                         lookup=chunk[LOOKUP],
                         p_arrival=chunk[P_ARRIVAL])
        self._chunk = chunk
        self.x_offer = None
        self.x_thread = None
        self.inputs = None

    def next_lstg(self):
        super().next_lstg()
        self.x_offer = subset_df(df=self._chunk[X_OFFER], lstg=self.lstg)
        self.x_thread = subset_df(df=self._chunk[X_THREAD], lstg=self.lstg)
        self.inputs = subset_inputs(input_data=self._chunk['inputs'],
                                    level='lstg', value=self.lstg)
        return self.x_lstg, self.lookup, self.p_arrival
