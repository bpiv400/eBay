from rlenv.LstgLoader import ChunkLoader
from utils import subset_df
from test.util import subset_inputs


class TestLoader(ChunkLoader):
    def __init__(self, x_lstg=None, lookup=None, test_data=None):
        super().__init__(x_lstg=x_lstg, lookup=lookup)
        self._test_data = test_data
        self.x_offer = None
        self.x_thread = None
        self.inputs = None

    def next_lstg(self):
        super().next_lstg()
        self.x_offer = subset_df(df=self._test_data['x_offer'],
                                 lstg=self.lstg)
        self.x_thread = subset_df(df=self._test_data['x_offer'],
                                  lstg=self.lstg)
        self.lstg = subset_inputs(input_data=self._test_data['inputs'],
                                  level='lstg', value=self.lstg)