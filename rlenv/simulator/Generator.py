"""
Generates values or discrim inputs for a chunk of lstgs
"""
import os
import sys
import shutil
from datetime import datetime
from compress_pickle import load, dump
from pympler import muppy, summary
from pympler.garbagegraph import GarbageGraph, start_debug_garbage
import gc
import numpy as np
from rlenv.env_consts import (VERBOSE, START_PRICE, START_DAY,
                              ACC_PRICE, DEC_PRICE, SILENT, MAX_RECORDER_SIZE)
from rlenv.interface.interfaces import PlayerInterface, ArrivalInterface
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.composer.Composer import Composer


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
        checkpoint_contents: the dictionary that the generated loaded from a checkpoint file
        checkpoint_count: The number of checkpoints saved for this chunk so far
        recorder_count: The number of recorders stored for this chunk, including the current one
        that hasn't been saved yet
        start: time that the current iteration of RewardGenerator began
        has_checkpoint: whether a recent checkpoint file has been created for this environment
    Public Functions:
        generate: generates values or discrim inputs based on the lstgs in the current chunk
    Static Methods:
        remake_dir: deletes and regenerates a given directory
        header: prints a header giving basic info for a lstg
    """
    def __init__(self, direct=None, num=None):
        """
        Constructor
        :param direct: string giving path to directory for current partition in rewardGenerator
        :param num: int giving the chunk number
        """
        start_debug_garbage()
        self.dir = direct
        self.chunk = int(num)
        input_path = '{}chunks/{}.gz'.format(self.dir, self.chunk)
        input_dict = load(input_path)
        self.x_lstg = input_dict['x_lstg']
        self.lookup = input_dict['lookup']
        self.lookup[START_DAY] = self.lookup[START_DAY].astype(np.int64) * 3600 * 24
        self.recorder = None
        composer = Composer(rebuild=False)
        self.buyer = PlayerInterface(composer=composer, byr=True)
        self.seller = PlayerInterface(composer=composer, byr=False)
        self.arrival = ArrivalInterface(composer=composer)

        # checkpoint params
        self.checkpoint_contents = None
        self.checkpoint_count = 0
        self.recorder_count = 1
        self.start = datetime.now()
        self.has_checkpoint = self._has_checkpoint()
        if not SILENT:
            print('checkpoint: {}'.format(self.has_checkpoint))
            print('checkpoint count: {}'.format(self.checkpoint_count))

        # delete previous directories for simulator and records
        if not self.has_checkpoint:
            self._prepare_records()

        # memory debugging
        self.prev_mem = summary.summarize(muppy.get_objects())

    def _prepare_records(self):
        """
        Clears existing value and record directories for this chunk
        """
        self._remake_dir(self.records_path)

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

    def mem_check(self):
        gc.collect()
        curr_mem = summary.summarize(muppy.get_objects())
        diff = summary.get_diff(self.prev_mem, curr_mem)
        self.prev_mem = curr_mem
        print('new summary')
        print('')
        summary.print_(diff)
        print('generator size: {}'.format(sys.getsizeof(self)))
        sys.stdout.flush()

    def setup_env(self, lstg, lookup):
        """
        Generates the environment required to
        simulate the given listing
        :param lstg: int giving a lstg id
        :param pandas.Series lookup: metadata about lstg
        :return: RewardEnvironment
        """
        # setup new lstg
        x_lstg = self.x_lstg.loc[lstg, :].astype(np.float32)
        environment = SimulatorEnvironment(buyer=self.buyer, seller=self.seller,
                                           arrival=self.arrival, x_lstg=x_lstg,
                                           lookup=lookup, recorder=self.recorder)
        if self.checkpoint_contents is None:
            # seed the recorder with a new lstg and make a new value calculator
            self.recorder.update_lstg(lookup=lookup, lstg=lstg)
        else:
            self.checkpoint_contents = None
        return environment

    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        remaining_lstgs = self._get_lstgs()
        time_up = False
        for lstg in remaining_lstgs:
            lookup = self.lookup.loc[lstg, :]
            environment = self.setup_env(lstg, lookup)
            # simulate lstg necessary number of times
            Generator.header(lstg, lookup)
            time_up = self.simulate_lstg_loop(environment)
            # store a checkpoint if the job is about to be killed
            if time_up:
                self._memory_dump()
                self.store_checkpoint(lstg)
                break
            else:
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
        self._generate_done()

    def _generate_done(self):
        path = '{}done_{}.txt'.format(self.records_path, self.chunk)
        open(path, 'a').close()

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
        path = self.checkpoint_path
        if os.path.isfile(path):
            os.remove(path)

    def store_checkpoint(self, lstg):
        """
        Creates a checkpoint for the current progress of the RewardGenerator,
        so the job can be killed and restarted in the short queue
        :param lstg: int giving lstg id
        :return: None
        """
        self.checkpoint_count += 1
        contents = self.make_checkpoint(lstg)
        path = self.checkpoint_path
        dump(contents, path)

    def simulate_lstg_loop(self, environment):
        raise NotImplementedError()

    def make_checkpoint(self, lstg):
        contents = {
            'lstg': lstg,
            'recorder': self.recorder,
            'recorder_count': self.recorder_count,
            'checkpoint_count': self.checkpoint_count,
            'time': datetime.now()
        }
        return contents

    def simulate_lstg(self, environment):
        """
        Simulates a particular listing once
        :param environment: RewardEnvironment
        :return: 2-tuple containing outcome tuple and boolean indicating whether the job
        has run out of queue time
        """
        environment.reset()
        outcome = environment.run()
        self._print_sim()
        time_up = self._check_time()
        self.recorder.add_sale(*outcome)
        return outcome, time_up

    def _check_time(self):
        """
        Checks whether the generator has been running for almost 4 hours
        :return: Boolean indicating almost 4 hours has passed
        """
        curr = datetime.now()
        tot = (curr - self.start).total_seconds() / 3600
        return tot > .25

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
        if sys.getsizeof(self.recorder) > MAX_RECORDER_SIZE:
            self.recorder.dump(self.recorder_count)
            self.recorder = self.make_recorder()
            self.recorder_count += 1

    @staticmethod
    def _memory_dump():
        print("FULL MEMORY DUMP")
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)
        gb = GarbageGraph(reduce=True)
        gb.render('garbage.eps')

    def _has_checkpoint(self):
        """
        Checks whether there's a recent checkpoint for the current chunk.
        Loads the checkpoint if one exists
        :returns: Indicator giving whether there's a recent checkpoint
        """
        path = self.checkpoint_path

        if os.path.isfile(self.checkpoint_path):
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

    @property
    def checkpoint_path(self):
        raise NotImplementedError()

    @property
    def records_path(self):
        raise NotImplementedError()

    def make_recorder(self):
        raise NotImplementedError()

    @staticmethod
    def header(lstg, lookup):
        """
        Prints header giving basic info about the current lstg
        :param lstg: int giving lstg id
        :param lookup: pd.Series containing metadata about the lstg
        :return:
        """
        if not SILENT:
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


