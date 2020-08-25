from datetime import datetime as dt
from rlenv.generate.Recorder import OutcomeRecorder
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from rlenv.LstgLoader import ChunkLoader
from rlenv.QueryStrategy import DefaultQueryStrategy
from rlenv.util import get_env_sim_dir, load_chunk


class Generator:
    def __init__(self, verbose=False, byr_agent=False):
        """
        Constructor
        :param bool verbose: if True, print info about simulator activity
        :param bool byr_agent: if True, flags agent buyer thread id in recorder
        """
        self.verbose = verbose
        self.byr_agent = byr_agent
        self.initialized = False

        # model interfaces and input composer
        self.recorder = None
        self.composer = None
        self.loader = None
        self.query_strategy = None
        self.env = None

    def process_chunk(self, part=None, chunk=None):
        self.loader = self.load_chunk(part=part, chunk=chunk)
        if not self.initialized:
            self.initialize()
        self.env = self.generate_env()
        return self.generate()

    def initialize(self):
        self.composer = self.generate_composer()
        self.query_strategy = self.generate_query_strategy()
        self.recorder = self.generate_recorder()
        self.initialized = True

    def generate_recorder(self):
        return OutcomeRecorder(verbose=self.verbose,
                               byr_agent=self.byr_agent)

    def generate_composer(self):
        raise NotImplementedError()

    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        print('Total listings: {}'.format(len(self.loader)))
        t0 = dt.now()
        while self.env.has_next_lstg():
            self.env.next_lstg()
            self.simulate_lstg()

        # time elapsed
        print('Avg time per listing: {} seconds'.format(
            (dt.now() - t0).total_seconds() / len(self.loader)))

        # return a dictionary
        return self.recorder.construct_output()

    def simulate_lstg(self):
        raise NotImplementedError()

    @property
    def env_class(self):
        raise NotImplementedError

    def generate_env(self):
        return self.env_class(query_strategy=self.query_strategy,
                              loader=self.loader,
                              recorder=self.recorder,
                              verbose=self.verbose,
                              composer=self.composer)

    def generate_buyer(self):
        return SimulatedBuyer(full=True)

    def generate_seller(self):
        return SimulatedSeller(full=True)

    def generate_query_strategy(self):
        buyer = self.generate_buyer()
        seller = self.generate_seller()
        arrival = ArrivalInterface()
        return DefaultQueryStrategy(buyer=buyer,
                                    seller=seller,
                                    arrival=arrival)

    def load_chunk(self, part=None, chunk=None):
        base_dir = get_env_sim_dir(part=part)
        x_lstg, lookup, p_arrival = load_chunk(base_dir=base_dir,
                                               num=chunk)
        return ChunkLoader(x_lstg=x_lstg,
                           lookup=lookup,
                           p_arrival=p_arrival)
