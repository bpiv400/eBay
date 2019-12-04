"""

"""
from compress_pickle import load
import torch
import pandas as pd
import numpy as np
from rlenv.env_consts import (REWARD_EXPERIMENT_PATH, SIM_COUNT, VERBOSE,
                              START_PRICE, START_DAY, ACC_PRICE, DEC_PRICE)
from rlenv.interface.interfaces import PlayerInterface, ArrivalInterface
from rlenv.rewards.RewardEnvironment import RewardEnvironment
from rlenv.composer.Composer import Composer
from rlenv.Recorder import Recorder


class RewardGenerator:
    """
    Attributes:
        exp_id: int giving the id of the current experiment
        params: dictionary containing parameters of the experiment:
            SIM_COUNT: number of times environment should simulator each lstg
            model in MODELS: integer giving the experiment of id for each model to be used in simulator
    """
    def __init__(self, direct, num, exp_id):
        super(RewardGenerator, self).__init__()
        self.dir = direct
        self.chunk = int(num)
        input_dict = load('{}chunks/{}.gz'.format(self.dir, self.chunk))
        self.x_lstg = input_dict['x_lstg']
        self.lookup = input_dict['lookup']
        self.exp_id = int(exp_id)
        self.params = self._load_params()
        self.recorder = Recorder(chunk=self.chunk)
        composer = Composer(self.params)
        self.buyer = PlayerInterface(composer=composer, byr=True)
        self.seller = PlayerInterface(composer=composer, byr=False)
        self.arrival = ArrivalInterface(composer=composer)

    def _load_params(self):
        """
        Loads dictionary of parameters associated with the current experiment
        from experiments spreadsheet

        :return: dictionary containing parameter values
        """
        params = pd.read_csv(REWARD_EXPERIMENT_PATH)
        params.set_index('id', drop=True, inplace=True)
        params = params.loc[self.exp_id, :].to_dict()
        return params

    def generate(self):
        for lstg in self.x_lstg.index:
            x_lstg = self.x_lstg.loc[lstg, :].astype(np.float32)
            lookup = self.lookup.loc[lstg, :]
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)
            environment = RewardEnvironment(buyer=self.buyer, seller=self.seller,
                                            arrival=self.arrival, x_lstg=x_lstg,
                                            lookup=lookup, recorder=self.recorder)
            if VERBOSE:
                header = 'Lstg: {}  | Start time: {}  | Start price: {}'.format(lstg,
                                                                                lookup[START_DAY],
                                                                                lookup[START_PRICE])
                header = '{} | auto reject: {} | auto accept: {}'.format(header, lookup[DEC_PRICE],
                                                                         lookup[ACC_PRICE])
                print(header)

            for i in range(self.params[SIM_COUNT]):
                environment.reset()
                # TODO Add event tracker
                sale, reward, time = environment.run()
                if VERBOSE:
                    print('Simulation {} concluded'.format(self.recorder.sim))
                self.recorder.add_sale(sale, reward, time)

            self.recorder.dump(self.dir)
