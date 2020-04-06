"""
Generates values or discrim inputs for a chunk of lstgs

If a checkpoint file for the current simulation, we load it and pick up where we left off.
Otherwise, starts from scratch with the first listing in the file

Lots of memory dumping code while I try to find leak
"""
import numpy as np
from featnames import START_PRICE, START_TIME, ACC_PRICE, DEC_PRICE
from rlenv.env_utils import load_chunk
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from rlenv.Composer import Composer
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment


class Generator:
    def __init__(self, direct, num, verbose=False):
        """
        Constructor
        :param direct: base directory for current partition
        :param int num:  chunk number
        :param verbose: whether to print info about simulator activity
        """
        self.dir = direct
        self.chunk = int(num)
        self.verbose = verbose
        self.x_lstg, self.lookup = load_chunk(base_dir=self.dir, num=self.chunk)
        self.recorder = None

        # model interfaces and input composer
        self.composer = self.generate_composer()
        self.buyer = self.generate_buyer()
        self.seller = self.generate_seller()
        self.arrival = ArrivalInterface()

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

        # index x_lstg
        x_lstg = self.x_lstg.loc[lstg, :].astype(np.float32)
        x_lstg = self.composer.decompose_x_lstg(x_lstg)
        # create and return environment
        return self.create_env(x_lstg=x_lstg, lookup=lookup, log=log)

    def create_env(self, x_lstg=None, lookup=None, log=None):
        return SimulatorEnvironment(buyer=self.buyer, seller=self.seller,
                                    arrival=self.arrival, x_lstg=x_lstg,
                                    lookup=lookup, recorder=self.recorder,
                                    verbose=self.verbose, composer=self.composer)

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

    def generate(self):
        raise NotImplementedError()

    def simulate_lstg(self, environment):
        raise NotImplementedError()

    @property
    def records_path(self):
        raise NotImplementedError()
