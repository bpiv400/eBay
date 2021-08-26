import pandas as pd
from env.LstgLoader import ChunkLoader
from testing.util import subset_inputs
from featnames import X_LSTG, X_OFFER, X_THREAD, LOOKUP, LSTG


class TestLoader(ChunkLoader):
    def __init__(self, chunk=None):
        super().__init__(x_lstg=chunk[X_LSTG], lookup=chunk[LOOKUP])
        self._chunk = chunk
        self.x_offer = None
        self.x_thread = None
        self.inputs = None

    def next_lstg(self):
        super().next_lstg()
        self.x_offer = self.subset_df(df=self._chunk[X_OFFER])
        self.x_thread = self.subset_df(df=self._chunk[X_THREAD])
        self.inputs = subset_inputs(input_data=self._chunk['inputs'],
                                    level=LSTG, value=self.lstg)
        return self.x_lstg, self.lookup, self.arrivals

    def subset_df(self, df=None):
        """
        Subsets an arbitrary dataframe to only contain rows for the given lstg
        :param df: pd.DataFrame
        :return: pd.Series or pd.DataFrame
        """
        df = df.copy()
        if isinstance(df.index, pd.MultiIndex):
            lstgs = df.index.unique(level=LSTG)
            if self.lstg in lstgs:
                return df.xs(self.lstg, level=LSTG, drop_level=True)
        else:
            if self.lstg in df.index:
                return df.loc[self.lstg, :]
        return None
