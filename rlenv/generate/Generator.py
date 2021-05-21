from datetime import datetime as dt
import pandas as pd
from rlenv.Composer import Composer
from rlenv.EBayEnv import EBayEnv
from rlenv.generate.Recorder import OutcomeRecorder
from rlenv.Player import SimulatedSeller, SimulatedBuyer
from rlenv.LstgLoader import ChunkLoader
from rlenv.QueryStrategy import DefaultQueryStrategy
from rlenv.util import load_chunk
from featnames import LSTG, SIM


class Generator:
    def __init__(self, verbose=False, test=False):
        """
        Constructor
        :param bool verbose: if True, print info about simulator activity
        :param bool test: if True, does not advance listing
        """
        self.verbose = verbose
        self.initialized = False
        self.test = test

        # model interfaces and input composer
        self.recorder = None
        self.composer = None
        self.loader = None
        self.query_strategy = None
        self.env = None

    def process_chunk(self, part=None, chunk=None, num_sims=1):
        self.loader = self.load_chunk(part=part, chunk=chunk, num_sims=num_sims)
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
        return None

    def generate_composer(self):
        return Composer()

    def generate(self):
        raise NotImplementedError()

    def simulate_lstg(self):
        """
        Simulates listing once.
        :return: None
        """
        self.env.reset()
        self.env.run()

    @property
    def env_class(self):
        return EBayEnv

    @property
    def env_args(self):
        return dict(query_strategy=self.query_strategy,
                    loader=self.loader,
                    recorder=self.recorder,
                    verbose=self.verbose,
                    composer=self.composer,
                    test=self.test)

    def generate_env(self):
        return self.env_class(**self.env_args)

    @staticmethod
    def generate_buyer():
        return SimulatedBuyer(full=True)

    @staticmethod
    def generate_seller():
        return SimulatedSeller(full=True)

    def generate_query_strategy(self):
        buyer = self.generate_buyer()
        seller = self.generate_seller()
        return DefaultQueryStrategy(buyer=buyer, seller=seller)

    def load_chunk(self, part=None, chunk=None, num_sims=None):
        x_lstg, lookup, arrivals = load_chunk(part=part, num=chunk)
        return ChunkLoader(x_lstg=x_lstg,
                           lookup=lookup,
                           arrivals=arrivals,
                           num_sims=num_sims)


class OutcomeGenerator(Generator):

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

    def generate_recorder(self):
        return OutcomeRecorder(verbose=self.verbose)


class ValueGenerator(Generator):

    def generate(self):
        print('Total listings: {}'.format(len(self.loader)))
        t0 = dt.now()

        rows = []
        while self.env.has_next_lstg():
            lstg, sim = self.env.next_lstg()
            self.simulate_lstg()
            rows.append([lstg, sim, self.env.outcome.price])

        # time elapsed
        print('Avg time per listing: {} seconds'.format(
            (dt.now() - t0).total_seconds() / len(self.loader)))

        # return series of sale prices
        df = pd.DataFrame.from_records(rows, columns=[LSTG, SIM, 'sale_price'])
        s = df.set_index([LSTG, SIM]).sort_index().squeeze()
        return s
