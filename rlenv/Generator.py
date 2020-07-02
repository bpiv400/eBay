"""
Generates values or discrim inputs for a chunk of lstgs


"""
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.environments.EbayEnvironment import EbayEnvironment
from rlenv.Composer import Composer
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from rlenv.DefaultQueryStrategy import DefaultQueryStrategy


class Generator:
    def __init__(self, part=None, verbose=False):
        """
        Constructor
        :param str part: name of partition
        :param bool verbose: if True, print info about simulator activity
        """
        self.part = part
        self.verbose = verbose
        self.initialized = False

        # model interfaces and input composer
        self.recorder = None
        self.composer = None
        self.loader = None
        self.query_strategy = None
        self.environment = None  # type: EbayEnvironment

    def process_chunk(self, chunk=None):
        self.loader = self.load_chunk(chunk=chunk)
        if not self.initialized:
            self.initialize()
        self.environment = self.generate_environment()
        return self.generate()

    def initialize(self):
        self.composer = self.generate_composer()
        self.query_strategy = self.generate_query_strategy()
        self.recorder = self.generate_recorder()
        self.initialized = True

    def load_chunk(self, chunk=None):
        raise NotImplementedError()

    def generate_recorder(self):
        raise NotImplementedError()

    def generate_query_strategy(self):
        raise NotImplementedError()

    def generate_composer(self):
        raise NotImplementedError()

    def generate(self):
        raise NotImplementedError()

    @property
    def env_class(self):
        raise NotImplementedError

    def generate_environment(self):
        return self.env_class(query_strategy=self.query_strategy,
                              loader=self.loader,
                              recorder=self.recorder,
                              verbose=self.verbose,
                              composer=self.composer)

    @property
    def records_path(self):
        """
        Returns path to the output directory for this simulation chunk
        :return: str
        """
        raise NotImplementedError()


class SimulatorGenerator(Generator):
    def generate_composer(self):
        return Composer(cols=self.loader.x_lstg_cols)

    def generate_query_strategy(self):
        buyer = self.generate_buyer()
        seller = self.generate_seller()
        arrival = ArrivalInterface()
        return DefaultQueryStrategy(buyer=buyer,
                                    seller=seller,
                                    arrival=arrival)

    @property
    def env_class(self):
        return SimulatorEnvironment

    def generate_buyer(self):
        return SimulatedBuyer()

    def generate_seller(self):
        return SimulatedSeller(full=True)

    def generate_recorder(self):
        raise NotImplementedError()

    def load_chunk(self, chunk=None):
        raise NotImplementedError()

    def generate(self):
        raise NotImplementedError()

    @property
    def records_path(self):
        raise NotImplementedError()
