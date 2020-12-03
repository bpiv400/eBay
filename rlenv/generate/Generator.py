from datetime import datetime as dt
import pandas as pd
from rlenv.Composer import Composer
from rlenv.generate.Recorder import OutcomeRecorder
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from rlenv.LstgLoader import ChunkLoader
from rlenv.QueryStrategy import DefaultQueryStrategy
from rlenv.util import get_env_sim_dir, load_chunk
from constants import OUTCOME_SIMS, VALUE_SIMS
from featnames import LSTG


VALUE_COLS = [LSTG, 'sale', 'sale_price', 'relist_ct']


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
        return OutcomeRecorder(verbose=self.verbose)

    def generate_composer(self):
        return Composer()

    def generate(self):
        raise NotImplementedError()

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
                              composer=self.composer,
                              test=self.test)

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


class OutcomeGenerator(Generator):

    def __init__(self, env):
        super().__init__(verbose=False, test=True)
        self.env = env

    @property
    def env_class(self):
        return self.env

    def simulate_lstg(self):
        """
        Simulates listing once.
        :return: None
        """
        self.env.reset()
        self.env.run()

    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        print('Total listings: {}'.format(len(self.loader)))
        t0 = dt.now()
        while self.env.has_next_lstg():
            self.env.next_lstg()
            for i in range(OUTCOME_SIMS):
                self.simulate_lstg()
                if self.recorder is not None:
                    self.recorder.increment_sim()

        # time elapsed
        print('Avg time per listing: {} seconds'.format(
            (dt.now() - t0).total_seconds() / len(self.loader)))

        # return a dictionary
        if self.recorder is not None:
            return self.recorder.construct_output()


class ValueGenerator(OutcomeGenerator):

    def generate_recorder(self):
        return None

    def generate(self):
        rows = []
        while self.env.has_next_lstg():
            start = dt.now()
            lstg = self.env.next_lstg()
            for i in range(VALUE_SIMS):
                row = self.simulate_lstg()
                rows.append([lstg, i] + list(row))

            # print listing summary
            elapsed = int(round((dt.now() - start).total_seconds()))
            print('{}: {} sec'.format(lstg, elapsed))

        # convert to dataframe
        df = pd.DataFrame.from_records(rows, columns=VALUE_COLS)
        df = df.set_index([LSTG, 'sale']).sort_index()

        # collapse
        x = df.sale_price.groupby(LSTG).mean()
        num_sales = df.relist_ct.groupby(LSTG).count()
        num_exps = df.relist_ct.groupby(LSTG).sum()
        p = num_sales / (num_sales + num_exps)
        values = pd.concat([x.rename('x'), p.rename('p')], axis=1).sort_index()

        return values

    def simulate_lstg(self):
        """
        Simulate until sale.
        :return: (sale price, relist count)
        """
        relist_ct = -1
        while relist_ct < 1000:
            relist_ct += 1
            self.env.reset()
            self.env.run()
            if self.env.outcome.sale:
                break
        return self.env.outcome.price, relist_ct
