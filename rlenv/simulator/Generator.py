"""
Generates values or discrim inputs for a chunk of lstgs

If a checkpoint file for the current simulation, we load it and pick up where we left off.
Otherwise, starts from scratch with the first listing in the file

Lots of memory dumping code while I try to find leak
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
from rlenv.env_consts import (VERBOSE, START_PRICE, START_DAY, RECORDER_DUMP_WIDTH,
                              ACC_PRICE, DEC_PRICE, SILENT, MAX_RECORDER_SIZE)
from rlenv.env_utils import load_chunk, get_done_file
from rlenv.interface.interfaces import PlayerInterface, ArrivalInterface
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.composer.Composer import Composer


class Generator:
    def __init__(self, direct=None, num=None):
        """
        Constructor
        :param direct: base directory for current partition
        :param int num:  chunk number
        """
        start_debug_garbage()
        self.start = datetime.now()
        self.dir = direct
        self.chunk = int(num)

        # load inputs
        self.x_lstg, self.lookup = load_chunk(self.dir, num)

        # interfaces for the models
        composer = Composer(rebuild=False)
        self.buyer = PlayerInterface(composer=composer, byr=True)
        self.seller = PlayerInterface(composer=composer, byr=False)
        self.arrival = ArrivalInterface(composer=composer)

        # checkpoint params -- default to no checkpoint
        self.checkpoint_contents = self._load_checkpoint()
        if self.checkpoint_contents is not None:
            self.checkpoint_count = self.checkpoint_contents['checkpoint_count']
            self.recorder_count = self.checkpoint_contents['recorder_count']
            self.recorder = self.checkpoint_contents['recorder']
        else:
            self.checkpoint_count = 0
            self.recorder_count = 1
            self.recorder = None

        if not SILENT:
            print('checkpoint: {}'.format(self.checkpoint_contents is not None))
            print('checkpoint count: {}'.format(self.checkpoint_count))

        # delete previous directories for simulator and records
        if self.checkpoint_contents is None:
            self._remake_dir(self.records_path)
        # memory debugging
        self.prev_mem = summary.summarize(muppy.get_objects())

    def _get_lstgs(self):
        """
        Generates list of listings in the current chunk that haven't had outputs generated
        yet.

        If there is a checkpoint file, the list contains all lstgs that weren't
        simulated to completion in the previous iterations of Generator. Otherwise,
        return all lstgs
        :return: list
        """
        if self.checkpoint_contents is None:
            return list(self.x_lstg.index)
        else:
            start_lstg = self.checkpoint_contents['lstg']
            index = list(self.x_lstg.index)
            start_ix = index.index(start_lstg)
            index = index[start_ix:]
            return index

    def mem_check(self):
        """
        Runs garbage collection and prints information about the program memory
        """
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
        Generates the environment required to simulate the given listing
        :param lstg: int giving a lstg id
        :param pd.Series lookup: metadata about lstg
        :return: SimulatorEnvironment
        """
        # setup new lstg
        x_lstg = self.x_lstg.loc[lstg, :].astype(np.float32)
        environment = SimulatorEnvironment(buyer=self.buyer, seller=self.seller,
                                           arrival=self.arrival, x_lstg=x_lstg,
                                           lookup=lookup, recorder=self.recorder)
        if self.checkpoint_contents is None:
            # seed the recorder with a new lstg if not loading from a checkpoint
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
        path = get_done_file(self.records_path, self.chunk)
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
        """
        self.checkpoint_count += 1
        contents = self.make_checkpoint(lstg)
        path = self.checkpoint_path
        dump(contents, path)

    def simulate_lstg_loop(self, environment):
        """
        Abstract method: simulates a listing until time runs out or until
        stop condition is met
        :param environment: SimulatorEnvironment
        :return: bool giving whether time has run out
        """
        raise NotImplementedError()

    def make_checkpoint(self, lstg):
        """
        Generates a checkpoint for this simulation containing
        the lstg, current recorder, recorder counter, checkpoint counter
        :param int lstg: lstg id
        :return: dict
        """
        contents = {
            'lstg': lstg,
            'recorder': self.recorder,
            'recorder_count': self.recorder_count,
            'checkpoint_count': self.checkpoint_count,
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
        Checks whether the generator has been running for max tim
        :return: bool
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
        contains some minimum amount of data
        """
        if len(self.recorder) % RECORDER_DUMP_WIDTH == 0:
            if sys.getsizeof(self.recorder) > MAX_RECORDER_SIZE:
                self.recorder.dump(self.recorder_count)
                self.recorder = self.make_recorder()
                self.recorder_count += 1

    @staticmethod
    def _memory_dump():
        """
        Prints memory summary and builds garbage reference cycle graph
        """
        print("FULL MEMORY DUMP")
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)
        gb = GarbageGraph(reduce=True)
        gb.render('garbage.eps')

    def _load_checkpoint(self):
        """
        Checks whether there's a recent checkpoint for the current chunk.
        Loads the checkpoint if one exists
        :return dict of checkpoint_contents
        """
        path = self.checkpoint_path
        if os.path.isfile(self.checkpoint_path):
            return load(path)
        else:
            return None

    @property
    def checkpoint_path(self):
        """
        Gets path to the checkpoint file for this simulation chunk
        :return: str
        """
        raise NotImplementedError()

    @property
    def records_path(self):
        """
        Returns path to the output directory for this simulation chunk
        :return: str
        """
        raise NotImplementedError()

    def make_recorder(self):
        """
        Abstract method that creates a recorder for the Generator
        :return: rlenv.simulator.Recorder
        """
        raise NotImplementedError()

    @staticmethod
    def header(lstg, lookup):
        """
        Prints header giving basic info about the current lstg
        :param lstg: int giving lstg id
        :param lookup: pd.Series containing metadata about the lstg
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


