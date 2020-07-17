from datetime import datetime as dt
from rlenv.generate.Recorder import OutcomeRecorder
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from rlenv.LstgLoader import ChunkLoader
from rlenv.util import get_env_sim_dir, load_chunk
from rlenv.DefaultQueryStrategy import DefaultQueryStrategy
from rlenv.Composer import Composer


class Generator:
    def __init__(self, verbose=False):
        """
        Constructor
        :param bool verbose: if True, print info about simulator activity
        """
        self.verbose = verbose
        self.initialized = False

        # model interfaces and input composer
        self.recorder = None
        self.composer = None
        self.loader = None
        self.query_strategy = None
        self.environment = None

    def process_chunk(self, part=None, chunk=None):
        print('Loading chunk {}...'.format(chunk))
        self.loader = self.load_chunk(part=part, chunk=chunk)
        print('Initializing environment {}...'.format(chunk))
        if not self.initialized:
            self.initialize()
        self.environment = self.generate_environment()
        print('Processing chunk {}...'.format(chunk))
        return self.generate()

    def initialize(self):
        print('Initializing composer...')
        self.composer = self.generate_composer()
        print('Initializing query strategy...')
        self.query_strategy = self.generate_query_strategy()
        print('Initializing recorder...')
        self.recorder = self.generate_recorder()
        self.initialized = True

    def generate_recorder(self):
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

    def generate_buyer(self):
        return SimulatedBuyer(full=True)

    def generate_seller(self):
        return SimulatedSeller(full=True)

    def generate_query_strategy(self):
        print('initializing buyer')
        buyer = self.generate_buyer()
        print('initializing seller')
        seller = self.generate_seller()
        print('initializing arrival interface')
        arrival = ArrivalInterface()
        return DefaultQueryStrategy(buyer=buyer,
                                    seller=seller,
                                    arrival=arrival)

    def load_chunk(self, part=None, chunk=None):
        base_dir = get_env_sim_dir(part=part)
        x_lstg, lookup, p_arrival = load_chunk(base_dir=base_dir,
                                               num=chunk)
        return ChunkLoader(x_lstg=x_lstg, lookup=lookup,
                           p_arrival=p_arrival)


class SimulatorGenerator(Generator):
    def generate_composer(self):
        raise NotImplementedError()

    @property
    def env_class(self):
        raise NotImplementedError()

    def generate_recorder(self):
        raise NotImplementedError()

    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        print('Total listings: {}'.format(len(self.loader)))
        t0 = dt.now()
        while self.environment.has_next_lstg():
            self.environment.next_lstg()

            # simulate lstg until sale
            self.simulate_lstg()

        # time elapsed
        print('Avg time per listing: {} seconds'.format(
            (dt.now() - t0).total_seconds() / len(self.loader)))

        # return a dictionary
        return self.recorder.construct_output()

    def simulate_lstg(self):
        raise NotImplementedError()


class DiscrimGenerator(SimulatorGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def generate_composer(self):
        return Composer(cols=self.loader.x_lstg_cols)

    def generate_recorder(self):
        return OutcomeRecorder(verbose=self.verbose,
                               record_sim=False)

    @property
    def env_class(self):
        return SimulatorEnvironment

    def simulate_lstg(self):
        """
        Simulates a particular listing once.
        :param env: SimulatorEnvironment
        :return: outcome tuple
        """
        self.environment.reset()
        outcome = self.environment.run()
        return outcome
