"""
Generates values or discriminator inputs for a chunk of lstgs
"""
import os
import sys
import shutil
from datetime import datetime
from statistics import mean, variance
from compress_pickle import load, dump
import pandas as pd
import numpy as np
from rlenv.ValueCalculator import ValueCalculator
from rlenv.env_consts import (REWARD_EXPERIMENT_PATH, VAL_SE_TOL, VERBOSE,
                              START_PRICE, START_DAY, ACC_PRICE, DEC_PRICE,
                              VAL_SE_CHECK, GEN_VALUES, SIM_COUNT)
from rlenv.env_utils import chunk_dir
from rlenv.interface.interfaces import PlayerInterface, ArrivalInterface
from rlenv.rewards.RewardEnvironment import RewardEnvironment
from rlenv.composer.Composer import Composer
from rlenv.Recorder import Recorder


class RewardGenerator:
    """
    Attributes:
        exp_id: int giving the id of the current experiment
        params: dictionary containing parameters of the experiment from rewardGenerator/experiments.csv:
            SIM_COUNT: number of times environment should simulator each lstg if we are generating discrim outputs
            VAL_SE_TOL: maximum value standard error tolerance
            VAL_SE_CHECK: The number of trials after which the standard error estimate is updated
            GEN_VALUES: boolean giving whether the environment is generating value estimates
        x_lstg: pd.Dataframe containing the x_lstg (x_w2v, x_slr, x_cat, x_cndtn, etc...) for each listing
        lookup: pd.Dataframe containing features of each listing used to control environment behavior and outputs
            (e.g. list price, auto reject price, auto accept price, meta category, etc..)
        recorder: Recorder object that stores environment outputs (e.g. discriminator features,
            events df, etc...)
        buyer: PlayerInterface containing buyer models
        seller: PlayerInterface containing seller models
        arrival: ArrivalInterface containing arrival and hist models
        checkpoint_contents: the dictionary that the generated loaded from a checkpoint file
        checkpoint_count: The number of checkpoints saved for this chunk so far
        recorder_count: The number of recorders stored for this chunk, including the current one
        that hasn't been saved yet
        start: time that the current iteration of RewardGenerator began
        has_checkpoint: whether a recent checkpoint file has been created for this environment
    Public Functions:
        generate: generates values or discriminator inputs based on the lstgs in the current chunk
    Static Methods:
        remake_dir: deletes and regenerates a given directory
        header: prints a header giving basic info for a lstg
    """
    def __init__(self, direct=None, num=None, exp_id=None):
        """
        Constructor
        :param direct: string giving path to directory for current partition in rewardGenerator
        :param num: int giving the chunk number
        :param exp_id: int giving the experiment number
        """
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

        # checkpoint params
        self.checkpoint_contents = None
        self.checkpoint_count = 0
        self.recorder_count = 1
        self.start = datetime.now()
        self.has_checkpoint = self._has_checkpoint()
        print('checkpoint: {}'.format(self.has_checkpoint))

        # delete previous directories for rewards and records
        if not self.has_checkpoint:
            self._prepare_records()

    def _prepare_records(self):
        """
        Clears existing value and record directories for this chunk
        """
        records_path = chunk_dir(self.dir, self.chunk, records=True,
                                 discrim=not self.params[GEN_VALUES])
        self._remake_dir(records_path)
        rewards_path = chunk_dir(self.dir, self.chunk, rewards=True)
        self._remake_dir(rewards_path)

    def _load_params(self):
        """
        Loads dictionary of parameters associated with the current experiment
        from experiments spreadsheet
        :return: dictionary containing parameter values
        """
        params = pd.read_csv(REWARD_EXPERIMENT_PATH)
        params.set_index('id', drop=True, inplace=True)
        params = params.loc[self.exp_id, :].to_dict()
        params[VAL_SE_TOL] = float(params[VAL_SE_TOL])
        params[VAL_SE_CHECK] = int(params[VAL_SE_CHECK])
        params[SIM_COUNT] = int(params[SIM_COUNT])
        print(params[GEN_VALUES])
        print(type(params[GEN_VALUES]))
        print('params')
        print(params)
        return params

    def _get_lstgs(self):
        """
        Generates list of listings in the current chunk that haven't had outputs generated
        yet. If there is a checkpoint file, the list contains all lstgs that weren't
        simulated to completion in the previous iterations of RewardGenerator
        :return: list
        """
        if not self.has_checkpoint:
            return self.x_lstg.index
        else:
            start_lstg = self.checkpoint_contents['lstg']
            index = list(self.x_lstg.index)
            start_ix = index.index(start_lstg)
            index = index[start_ix:]
            return index

    def _setup_env(self, lstg):
        """
        Generates the environment, lookup series, and value calculator required to
        simulate the given listing
        :param lstg: int giving a lstg id
        :return: 3-tuple containing RewardEnvironment, ValueCalculator, pd.Series
        """
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
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        remaining_lstgs = self._get_lstgs()
        time_up = False
        for lstg in remaining_lstgs:
            environment, val_calc, lookup = self._setup_env(lstg)
            # simulate lstg necessary number of times
            RewardGenerator.header(lstg, lookup)
            print('simulate lstg loop')
            print(self.params[GEN_VALUES])
            time_up = self._simulate_lstg_loop(environment, val_calc)
            # store a checkpoint if the job is about to be killed
            if time_up:
                self._store_checkpoint(lstg, val_calc)
                break
            else:
                if self.params[GEN_VALUES]:
                    self.recorder.add_val(val_calc.mean)
                self._tidy_recorder()
        if not time_up:
            self._close()

    def _close(self):
        """
        Dumps the latest recorder and deletes the latest checkpoint (if it exists),
        after all lstgs in the chunk have been simulated
        """
        self.recorder.dump(self.dir, self.recorder_count)
        self._delete_checkpoint()
        self._report_time()

    def _report_time(self):
        """
        Prints the total amount of time spent simulating all lstgs in the chunk
        """
        curr_clock = (datetime.now() - self.start).total_seconds() / 3600
        print('hours: {}'.format(self.checkpoint_count * 4 + curr_clock))

    def _delete_checkpoint(self):
        """
        Deletes the existing checkpoint
        """
        path = '{}chunks/{}_check.gz'.format(self.dir, self.chunk)
        if os.path.isfile(path):
            os.remove(path)

    def _store_checkpoint(self, lstg, val_calc):
        """
        Creates a checkpoint for the current progress of the RewardGenerator,
        so the job can be killed and restarted in the short queue
        :param lstg: int giving lstg id
        :param val_calc: ValueCalculator
        :return: None
        """
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

    def _simulate_lstg_loop(self, environment, val_calc):
        """
        Simulates a particular lstg the requisite number of times
        :param environment: RewardEnvironment initiated for current lstg
        :param val_calc: ValueCalculator
        :return: Boolean indicating whether the RewardGenerator job has run out of time
        """
        if self.params[GEN_VALUES]:
            print('gen values')
            return self._value_loop(environment, val_calc)
        else:
            return self._discrim_loop(environment, val_calc)

    def _discrim_loop(self, environment, val_calc):
        """
        Simulates a particular listing a given number of times and stores
        outputs required to train discrimator
        :param environment: RewardEnvironment
        :param val_calc: ValueCalculator
        :return: Boolean indicating whether the job has run out of queue time
        """
        time_up = False
        while val_calc.exp_count < self.params[SIM_COUNT] and not time_up:
            time_up = self._simulate_lstg(environment, val_calc)
        return time_up

    def _simulate_lstg(self, environment, val_calc):
        """
        Simulates a particular listing once
        :param environment: RewardEnvironment
        :param val_calc: ValueCalculator
        :return: Boolean indicating whether the job has run out of queue time
        """
        environment.reset()
        sale, price, dur = environment.run()
        self._print_sim()
        val_calc.add_outcome(sale, price)
        time_up = self._check_time()
        return time_up

    def _value_loop(self, environment, val_calc):
        """
        Simulates a particular listing until the value estimate se is beneath the given tolerance
        :param environment: RewardEnvironment
        :param val_calc: ValueCalculator
        :return: Boolean indicating whether the job has run out of queue time
        """
        stop, time_up = False, False
        while not stop and not time_up:
            time_up = self._simulate_lstg(environment, val_calc)
            stop = self._update_stop(val_calc)
            self._print_value(val_calc)
        return time_up

    def _print_value(self, val_calc):
        """
        Prints information about the most recent value calculation
        :param val_calc: ValueCalculator
        """
        if val_calc.exp_count % self.params[VAL_SE_CHECK] == 0:
            print('Trial: {}'.format(val_calc.exp_count))
            if len(val_calc.sales) == 0:
                print('No Sales')
            elif len(val_calc.sales) == 1:
                print('Only 1 sale')
            else:
                print('cut: {}'.format(val_calc.cut))
                print('rate of no sale: {}'.format(val_calc.p))
                price_header = 'sale count: {} | price mean: {}'.format(len(val_calc.sales),
                                                                        mean(val_calc.sales))
                price_header = '{} | price var: {}'.format(price_header, variance(val_calc.sales))
                print(price_header)
                print('mean of value: {} | mean standard error: {}'.format(val_calc.mean,
                                                                           val_calc.mean_se))
                print('Predicted trials remaining: {}'.format(val_calc.trials_until_stable))

    def _check_time(self):
        """
        Checks whether the generator has been running for almost 4 hours
        :return: Boolean indicating almost 4 hours has passed
        """
        curr = datetime.now()
        tot = (curr - self.start).total_seconds() / 3600
        return tot > 3.90

    def _update_stop(self, val_calc):
        """
        Checks whether a the generator may stop simulating a particular lstg
        based on whether the value estimate has stabilized
        :param val_calc: ValueCalculator
        :return: Boolean indicating whether to stop
        """
        counter = val_calc.exp_count
        if val_calc.has_sales and counter % self.params[VAL_SE_CHECK] == 0:
            return val_calc.stabilized
        else:
            return False

    def _print_sim(self):
        """
        Prints that the simulation has concluded if verbose
        """
        if VERBOSE:
            print('Simulation {} concluded'.format(self.recorder.sim))

    def _tidy_recorder(self):
        """
        Dumps the recorder and increments the recorder count if it
        contains at least a gig of data
        """
        if sys.getsizeof(self.recorder) > 1e9:
            self.recorder.dump(self.dir, self.recorder_count)
            self.recorder = Recorder(chunk=self.chunk)
            self.recorder_count += 1

    def _has_checkpoint(self):
        """
        Checks whether there's a recent checkpoint for the current chunk.
        Loads the checkpoint if one exists
        :returns: Indicator giving whether there's a recent checkpoint
        """
        path = '{}chunks/{}_check.gz'.format(self.dir, self.chunk)
        has = os.path.isfile(path)
        if has:
            self.checkpoint_contents = load(path)
            time = self.checkpoint_contents['time']
            since = (time - datetime.now()).total_seconds() / 3600
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
        """
        Prints header giving basic info about the current lstg
        :param lstg: int giving lstg id
        :param lookup: pd.Series containing metadata about the lstg
        :return:
        """
        header = 'Lstg: {}  | Start time: {}  | Start price: {}'.format(lstg,
                                                                        lookup[START_DAY],
                                                                        lookup[START_PRICE])
        header = '{} | auto reject: {} | auto accept: {}'.format(header, lookup[DEC_PRICE],
                                                                 lookup[ACC_PRICE])
        print(header)

    @staticmethod
    def _remake_dir(path):
        """
        Deletes and re-creates the given directory
        :param path: path to some directory
        """
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)
