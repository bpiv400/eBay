"""
Generates values or discrim inputs for a chunk of lstgs
"""
import os, sys, shutil
from datetime import datetime
from compress_pickle import load, dump
import numpy as np
from rlenv.env_consts import START_PRICE, START_TIME, ACC_PRICE, DEC_PRICE
from rlenv.interface.interfaces import PlayerInterface, ArrivalInterface
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.Composer import Composer


class Generator:
    """
    Attributes:
        x_lstg: pd.Dataframe containing the x_lstg (x_w2v, x_slr, x_cat, x_cndtn, etc...) for each listing
        lookup: pd.Dataframe containing features of each listing used to control environment behavior and outputs
            (e.g. list price, auto reject price, auto accept price, meta category, etc..)
        recorder: Recorder object that stores environment outputs (e.g. discrim features,
            events df, etc...)
        buyer: PlayerInterface containing buyer models
        seller: PlayerInterface containing seller models
        arrival: ArrivalInterface containing arrival and hist models
    Public Functions:
        generate: generates values or discrim inputs based on the lstgs in the current chunk
    """
    def __init__(self, direct, num, verbose=False):
        """
        Constructor
        :param direct: string giving path to directory for current partition in rewardGenerator
        :param num: int giving the chunk number
        """
        self.dir = direct
        self.chunk = int(num)
        self.verbose = verbose

        d = load('{}chunks/{}.gz'.format(self.dir, self.chunk))
        self.x_lstg = d['x_lstg']
        self.lookup = d['lookup']
        self.recorder = None

        composer = Composer(self.x_lstg.columns)
        self.buyer = PlayerInterface(composer=composer, byr=True)
        self.seller = PlayerInterface(composer=composer, byr=False)
        self.arrival = ArrivalInterface(composer=composer)


    def generate(self):
        raise NotImplementedError()


    def _setup_env(self, lstg, lookup):
        """
        Generates the environment required to
        simulate the given listing
        :param lstg: int giving a lstg id
        :param pandas.Series lookup: metadata about lstg
        :return: RewardEnvironment
        """
        if self.verbose:
            self.print_lstg_info(lstg, lookup)

        # index x_lstg
        x_lstg = self.x_lstg.loc[lstg, :].astype(np.float32)

        # create and return environment
        return SimulatorEnvironment(buyer=self.buyer, seller=self.seller,
                                    arrival=self.arrival, x_lstg=x_lstg, lookup=lookup,
                                    recorder=self.recorder, verbose=self.verbose)


    def _simulate_lstg(self):
        raise NotImplementedError()

    @property
    def records_path(self):
        raise NotImplementedError()

    @staticmethod
    def print_lstg_info(lstg, lookup):
        """
        Prints header giving basic info about the current lstg
        :param lstg: int giving lstg id
        :param lookup: pd.Series containing metadata about the lstg
        :return:
        """
        print('lstg: {} | start_time: {} | start_price: {} | auto_rej: {} | auto_acc: {}'.format(
            lstg, lookup[START_TIME], lookup[START_PRICE], lookup[DEC_PRICE], lookup[ACC_PRICE]))