from rlenv.LstgLoader import ChunkLoader
from utils import subset_df
from test.util import subset_inputs


class TestLoader(ChunkLoader):
    def __init__(self, x_lstg=None, lookup=None, test_data=None):
        super().__init__(x_lstg=x_lstg, lookup=lookup)
        self._test_data = test_data

    def x_offer(self):
        return subset_df(df=self._test_data['x_offer'],
                         lstg=self.lstg)

    def x_thread(self):
        return subset_df(df=self._test_data['x_offer'],
                         lstg=self.lstg)

    def inputs(self):
        return subset_inputs(models=list(self._test_data['inputs'].keys()),
                             input_data=self._test_data['inputs'],
                             level='lstg', value=self.lstg)