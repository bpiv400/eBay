from compress_pickle import load
from datetime import datetime as dt

from rlenv.generate.Recorder import OutcomeRecorder
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from rlenv.Composer import Composer
from rlenv.util import get_env_sim_subdir
from featnames import START_PRICE, START_TIME, ACC_PRICE, DEC_PRICE, \
    X_LSTG, LOOKUP, P_ARRIVAL


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

        # data
        self.chunk = None
        self.x_lstg = None
        self.lookup = None
        self.p_arrival = None

        # model interfaces and input composer
        self.recorder = None
        self.composer = None
        self.seller = None
        self.buyer = None
        self.arrival = None

    def process_chunk(self, chunk=None):
        self.load_chunk(chunk=chunk)
        if not self.initialized:
            self.initialize()
        return self.generate()

    def initialize(self):
        self.composer = self.generate_composer()
        self.buyer = self.generate_buyer()
        self.seller = self.generate_seller()
        self.arrival = ArrivalInterface()
        self.recorder = self.generate_recorder()
        self.initialized = True

    def load_chunk(self, chunk=None):
        raise NotImplementedError()

    def generate_recorder(self):
        raise NotImplementedError()

    def generate_composer(self):
        raise NotImplementedError()

    def generate_buyer(self):
        raise NotImplementedError()

    def generate_seller(self):
        raise NotImplementedError()

    def generate(self):
        raise NotImplementedError()

    def setup_env(self, lstg=None, lookup=None, log=None):
        """
        Generates the environment required to simulate the given listing
        :param lstg: int giving a lstg id
        :param pd.Series lookup: metadata about lstg
        :param log: optional LstgLog passed if testing environment
        :return: SimulatorEnvironment or subclass
        """
        if self.verbose:
            self.print_lstg_info(lstg, lookup)

        # index x_lstg and p_arrival
        x_lstg = self.x_lstg.loc[lstg, :]
        x_lstg = self.composer.decompose_x_lstg(x_lstg)
        p_arrival = self.p_arrival.loc[lstg, :].values

        # create and return environment
        return self.create_env(x_lstg=x_lstg,
                               lookup=lookup,
                               p_arrival=p_arrival,
                               log=log)

    def create_env(self, x_lstg=None, lookup=None, p_arrival=None, log=None):
        return SimulatorEnvironment(buyer=self.buyer,
                                    seller=self.seller,
                                    arrival=self.arrival,
                                    x_lstg=x_lstg,
                                    lookup=lookup,
                                    p_arrival=p_arrival,
                                    recorder=self.recorder,
                                    verbose=self.verbose,
                                    composer=self.composer)

    def simulate_lstg(self, environment):
        raise NotImplementedError()

    @property
    def records_path(self):
        """
        Returns path to the output directory for this simulation chunk
        :return: str
        """
        raise NotImplementedError()

    @staticmethod
    def print_lstg_info(lstg, lookup):
        """
        Prints header giving basic info about the current lstg
        :param lstg: int giving lstg id
        :param lookup: pd.Series containing metadata about the lstg
        """
        print('lstg: {} | start_time: {} | start_price: {} | auto_rej: {} | auto_acc: {}'.format(
            lstg, lookup[START_TIME], lookup[START_PRICE], lookup[DEC_PRICE], lookup[ACC_PRICE]))


class SimulatorGenerator(Generator):
    def generate_composer(self):
        return Composer(self.x_lstg.columns)

    def generate_buyer(self):
        return SimulatedBuyer()

    def generate_seller(self):
        return SimulatedSeller(full=True)

    def generate_recorder(self):
        raise NotImplementedError()

    def load_chunk(self, chunk=None):
        self.chunk = chunk
        chunk_dir = get_env_sim_subdir(part=self.part,
                                       chunks=True)
        d = load(chunk_dir + '{}.gz'.format(chunk))
        self.lookup = d[LOOKUP]
        self.x_lstg = d[X_LSTG]
        self.p_arrival = d[P_ARRIVAL]

    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        print('Total listings: {}'.format(len(self.x_lstg)))
        t0 = dt.now()
        for lstg in self.x_lstg.index:
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            env = self.setup_env(lstg=lstg, lookup=lookup)

            # update listing in recorder
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)

            # simulate lstg once
            self.simulate_lstg(env)

        # time elapsed
        print('Avg time per listing: {} seconds'.format(
            (dt.now() - t0).total_seconds() / len(self.x_lstg.index)))

        # save the recorder
        self.recorder.dump()

    def simulate_lstg(self, environment):
        raise NotImplementedError()

    @property
    def records_path(self):
        raise NotImplementedError()


class DiscrimGenerator(SimulatorGenerator):
    def __init__(self, part=None, verbose=False):
        super().__init__(part=part, verbose=verbose)

    def generate_recorder(self):
        return OutcomeRecorder(records_path=self.records_path,
                               verbose=self.verbose,
                               record_sim=False)

    def simulate_lstg(self, env):
        """
        Simulates a particular listing once.
        :param env: SimulatorEnvironment
        :return: outcome tuple
        """
        env.reset()
        outcome = env.run()
        return outcome

    @property
    def records_path(self):
        out_dir = get_env_sim_subdir(part=self.part,
                                     discrim=True)
        return out_dir + '{}.gz'.format(self.chunk)
