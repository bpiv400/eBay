"""

"""
import os
import sys
import shutil
from datetime import datetime
from compress_pickle import load, dump
import pandas as pd
import numpy as np
from rlenv.ValueCalculator import ValueCalculator
from rlenv.env_consts import (REWARD_EXPERIMENT_PATH, VAL_SE_TOL, VERBOSE,
                              START_PRICE, START_DAY, ACC_PRICE, DEC_PRICE,
                              VAL_SE_CHECK)
from rlenv.env_utils import chunk_dir
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
    def __init__(self, direct=None, num=None, exp_id=None):
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

        # defaults
        self.checkpoint_contents = None
        self.checkpoint_count = 0
        self.recorder_count = 1
        self.start = datetime.now()
        # load checkpoint if there is one
        self.has_checkpoint = self._has_checkpoint()

        # delete previous directories for rewards and records
        if not self.has_checkpoint:
            self._prepare_records()

    def _prepare_records(self):
        records_path = chunk_dir(self.dir, self.chunk, records=True)
        rewards_path = chunk_dir(self.dir, self.chunk, rewards=True)
        RewardGenerator.remake_dir(records_path)
        RewardGenerator.remake_dir(rewards_path)

    @staticmethod
    def remake_dir(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            os.mkdir(path)

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

    def _get_lstgs(self):
        if not self.has_checkpoint:
            return self.x_lstg.index
        else:
            start_lstg = self.checkpoint_contents['lstg']
            index = list(self.x_lstg.index)
            start_ix = index.index(start_lstg)
            index = index[start_ix:]
            return index

    def setup_env(self, lstg):
        # setup new lstg
        x_lstg = self.x_lstg.loc[lstg, :].astype(np.float32)
        lookup = self.lookup.loc[lstg, :]
        # if there's a checkpoint and this is the first lstg
        if self.checkpoint_contents is not None:
            # grab the value calculator and reset checkpoint
            val_calc = self.checkpoint_contents['val_calc']
            self.checkpoint_contents = None
        else:
            # otherwise seed the recorder with a new lstg and make a new value calculator
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)
            val_calc = ValueCalculator(self.params[VAL_SE_TOL], lookup)
        environment = RewardEnvironment(buyer=self.buyer, seller=self.seller,
                                        arrival=self.arrival, x_lstg=x_lstg,
                                        lookup=lookup, recorder=self.recorder)
        return environment, val_calc, lookup

    def generate(self):
        remaining_lstgs = self._get_lstgs()
        time_up = False
        for lstg in remaining_lstgs:
            environment, val_calc, lookup = self.setup_env(lstg)
            # simulate lstg
            RewardGenerator.header(lstg, lookup)
            time_up = self.simulate_lstg(environment, val_calc)
            if time_up:
                self.store_checkpoint(lstg, val_calc)
                break
            else:
                self.recorder.add_val(val_calc.mean)
                self.tidy_recorder()
        if not time_up:
            self.recorder.dump(self.dir, self.recorder_count)
            self.delete_checkpoint()
            self.report_time()

    def report_time(self):
        curr_clock = (datetime.now() - self.start).total_seconds() / 3600
        print('hours: {}'.format(self.checkpoint_count * 4 + curr_clock))

    def delete_checkpoint(self):
        path = '{}chunks/{}_check.gz'.format(self.dir, self.chunk)
        if os.path.isfile(path):
            os.remove(path)

    def store_checkpoint(self, lstg, val_calc):
        self.checkpoint_count += 1
        contents = {
            'lstg': lstg,
            'val_calc': val_calc,
            'recorder': self.recorder,
            'recorder_count': self.recorder_count,
            'checkpoint_count': self.checkpoint_count,
            'time': datetime.now()
        }
        path = '{}chunks/{}_check.gz'.format(self.dir, self.chunk)
        dump(contents, path)

    def simulate_lstg(self, environment, val_calc):
        # stopping criterion
        stop, time_up = False, False
        while not stop and not time_up:
            environment.reset()
            sale, price, dur = environment.run()
            self.print_sim()
            val_calc.add_outcome(sale, price)
            stop = self.update_stop(val_calc)
            time_up = self.check_time()
        return time_up

    def check_time(self):
        curr = datetime.now()
        tot = (curr - self.start).total_seconds() / 3600
        return tot > 3.90

    def update_stop(self, val_calc):
        counter = val_calc.exp_count
        if val_calc.has_sales and counter % self.params[VAL_SE_CHECK] == 0:
            return val_calc.stabilized
        else:
            if not val_calc.has_sales:
                print('Trials: {}  - NO SALES'.format(counter))
            else:
                stable = val_calc.trials_until_stable
                print('Trials: {} - Predicted additional trials: {}'.format(counter, stable))
            return False

    # check whether we should check value
    def print_sim(self):
        if VERBOSE:
            print('Simulation {} concluded'.format(self.recorder.sim))

    def tidy_recorder(self):
        if sys.getsizeof(self.recorder) > 1e9:
            self.recorder.dump(self.dir, self.recorder_count)
            self.recorder = Recorder(chunk=self.chunk)
            self.recorder_count += 1

    def _has_checkpoint(self):
        path = '{}chunks/{}_check.gz'.format(self.dir, self.chunk)
        has = os.path.isfile(path)
        if has:
            self.checkpoint_contents = load(path)
            time = self.checkpoint_contents['time']
            since = (time - datetime.now()) / 3600
            if since > 24:
                self.checkpoint_contents = None
                return False
            else:
                self.checkpoint_count = self.checkpoint_contents['checkpoint_count']
                self.recorder_count = self.checkpoint_contents['recorder_count']
                self.recorder = self.checkpoint_contents['recorder']
                return True
        else:
            return False

    @staticmethod
    def header(lstg, lookup):
        if VERBOSE:
            header = 'Lstg: {}  | Start time: {}  | Start price: {}'.format(lstg,
                                                                            lookup[START_DAY],
                                                                            lookup[START_PRICE])
            header = '{} | auto reject: {} | auto accept: {}'.format(header, lookup[DEC_PRICE],
                                                                     lookup[ACC_PRICE])
            print(header)




