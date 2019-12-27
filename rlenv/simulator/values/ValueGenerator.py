import os, shutil
from datetime import datetime as dt
from rlenv.env_consts import MIN_SALES
from rlenv.env_utils import get_checkpoint_path, get_chunk_dir
from rlenv.simulator.Generator import Generator
from rlenv.simulator.values.ValueCalculator import ValueCalculator
from rlenv.simulator.values.ValueRecorder import ValueRecorder


class ValueGenerator(Generator):
    """
    Uninherited attributes:
        checkpoint_contents: the dictionary that the generated loaded from a checkpoint file
        checkpoint_count: The number of checkpoints saved for this chunk so far
        recorder_count: The number of recorders stored for this chunk, including the current one
        that hasn't been saved yet
        start: time that the current iteration of RewardGenerator began
        has_checkpoint: whether a recent checkpoint file has been created for this environment
    """
    def __init__(self, direct, num, verbose=False):
        super(ValueGenerator, self).__init__(direct, num, verbose)
        self.val_calc = None  # type: ValueCalculator
        self.recorder = self._make_recorder()

        # checkpoint params
        self.checkpoint_contents = None
        self.checkpoint_count = 0
        self.recorder_count = 1
        self.start = dt.now()
        self.has_checkpoint = self._has_checkpoint()
        if self.verbose:
            print('checkpoint: {}'.format(self.has_checkpoint))
            print('checkpoint count: {}'.format(self.checkpoint_count))

        # delete previous directories for simulator and records
        if not self.has_checkpoint:
            self._remake_dir(self.records_path)


    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        for lstg in self._get_lstgs():
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            environment = self._setup_env(lstg, lookup)

            # grab or initialize value calculator and if necessary, seed recorder
            if self.checkpoint_contents is None:
                self.val_calc = ValueCalculator(lookup=lookup)
                self.recorder.update_lstg(lookup=lookup, lstg=lstg)
            else:
                self.val_calc = self.checkpoint_contents['val_calc']
                self.checkpoint_contents = None

            # simulate lstg until a stopping criterion is satisfied
            time_up = False
            while not self.val_calc.stabilized and not time_up:
                price, T = self._simulate_lstg(environment)
                self.val_calc.add_outcome(price, T)
                if self.verbose:
                    self._print_value()
                time_up = self._is_time_up()

            # save results to value calculator
            self.recorder.add_val(self.val_calc)

            # store a checkpoint if the job is about to be killed
            if time_up:
                self.checkpoint_count += 1
                dump(self._make_checkpoint(lstg), self.checkpoint_path)
                break

        # clean up and save
        if not time_up:
            self.recorder.dump(self.recorder_count)
            self._delete_checkpoint()
            self._generate_done()


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


    def _simulate_lstg(self, environment):
        """
        Simulates a particular listing until it sells
        :param environment: RewardEnvironment
        :return: 2-tuple containing outcome tuple and boolean indicating whether the job
        has run out of queue time
        """
        T = 1
        while True:
            environment.reset()
            sale, price, _ = environment.run()
            if sale:
                return price, T
            T += 1


    def _print_value(self):
        """
        Prints information about the most recent value calculation
        """
        print('Sale #{0:d}'.format(self.val_calc.num_sales))
        if self.val_calc.num_sales == 0:
            print('No Sales')
        else:
            print('sale rate: {0:.2f} | avg sale price: {1:.2f}'.format(
                self.val_calc.p_sale, self.val_calc.mean_x))
            print('value: {0:.2f} | se: {1:.2f}'.format(
                self.val_calc.value, self.val_calc.se))


    def _make_checkpoint(self, lstg):
        contents = {
            'lstg': lstg,
            'recorder': self.recorder,
            'recorder_count': self.recorder_count,
            'checkpoint_count': self.checkpoint_count,
            'time': dt.now(),
            'val_calc': self.val_calc
        }
        return contents


    def _make_recorder(self):
        return ValueRecorder(self.records_path, self.verbose)

    def _update_stop(self):
        """
        Checks whether a the generator may stop simulating a particular lstg
        based on whether the value estimate has stabilized
        :return: Boolean indicating whether to stop
        """
        if self.val_calc.has_sales:
            return self.val_calc.stabilized
        else:
            return False

    @property
    def checkpoint_path(self):
        return get_checkpoint_path(self.dir, self.chunk, discrim=False)

    @property
    def records_path(self):
        return get_chunk_dir(self.dir, self.chunk, discrim=False)

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
            since = (time - dt.now()).total_seconds() / 3600
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

    def _delete_checkpoint(self):
        """
        Deletes the existing checkpoint
        """
        path = self.checkpoint_path
        if os.path.isfile(path):
            os.remove(path)

    @staticmethod
    def _remake_dir(path):
        """
        Deletes and re-creates the given directory
        :param path: path to some directory
        """
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)

    def _is_time_up(self):
        """
        Checks whether the generator has been running for almost 4 hours
        :return: Boolean indicating almost 4 hours has passed
        """
        tot = (dt.now() - self.start).total_seconds() / 3600
        return tot > .25


    def _generate_done(self):
        path = '{}done_{}.txt'.format(self.records_path, self.chunk)
        open(path, 'a').close()