from rlenv.LstgLoader import ChunkLoader
from testing.util import subset_inputs, subset_df


class TestLoader(ChunkLoader):
    def __init__(self, x_lstg=None, lookup=None, test_data=None,
                 agent=False, p_arrival=None):
        super().__init__(x_lstg=x_lstg, lookup=lookup,
                         p_arrival=p_arrival)
        self._test_data = test_data
        self.x_offer = None
        self.x_thread = None
        self.inputs = None
        self.block_next = False
        self.agent = agent

    def has_next(self):
        return super().has_next() or self.block_next

    def next_lstg(self):
        if self.block_next and self.agent:
            self.block_next = False
        else:
            super().next_lstg()
            self.x_offer = subset_df(df=self._test_data['x_offer'],
                                     lstg=self.lstg)
            self.x_thread = subset_df(df=self._test_data['x_thread'],
                                      lstg=self.lstg)
            self.inputs = subset_inputs(input_data=self._test_data['inputs'],
                                        level='lstg', value=self.lstg)
            self.block_next = True
        return self.x_lstg, self.lookup, self.p_arrival
