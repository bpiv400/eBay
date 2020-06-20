class LstgLoader:

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


class ChunkLoader(LstgLoader):
    def next_id(self):
        pass

    def next_lstg(self):
        pass

    def has_next(self):
        pass

    def init(self):
        pass


class TrainLoader(LstgLoader):
    def next_lstg(self):
        pass

    def has_next(self):
        pass

    def next_id(self):
        pass

    def init(self):
        pass