import os
import sys
from datetime import datetime as dt
from compress_pickle import dump, load
from rlenv.util import get_checkpoint_path
from rlenv.const import SIM_VALS_DIR, VAL_TIME_LIMIT
from rlenv.generate.Generator import SimulatorGenerator
from agent.eval.ValueCalculator import ValueCalculator
from agent.eval.ValueRecorder import ValueRecorder


# TODO: Deprecate or update
class ValueGenerator(SimulatorGenerator):
    """
    Uninherited attributes:
        checkpoint_contents: the dictionary that the generated loaded from a checkpoint file
        checkpoint_count: The number of checkpoints saved for this chunk so far
        start: time that the current iteration of RewardGenerator began
        has_checkpoint: whether a recent checkpoint file has been created for this environment
    """

    #TODO: Implement
    def load_chunk(self, chunk=None):
        pass

    def __init__(self, verbose=False, start=None):
        super(ValueGenerator, self).__init__(verbose=verbose)
        self.val_calc = None  # type: ValueCalculator

        # checkpoint params
        self.checkpoint_contents = None
        self.checkpoint_count = 0
        self.has_checkpoint = False

        self.start = dt.now()

    def initialize(self):
        super().initialize()
        self.has_checkpoint = self._has_checkpoint()
        if self.verbose:
            print('checkpoint: {}'.format(self.has_checkpoint))
            print('checkpoint count: {}'.format(self.checkpoint_count))

    def generate(self):
        """
        Simulates all lstgs in chunk according to experiment parameters
        """
        time_up = False
        remaining_lstgs = self._get_lstgs()
        last_perc = 0.0
        # progress bar crap
        last_checkin = dt.now()
        for i, lstg in enumerate(remaining_lstgs):
            # index lookup dataframe
            lookup = self.lookup.loc[lstg, :]

            # create environment
            environment = self.setup_env(lstg=lstg, lookup=lookup)

            # grab or initialize value calculator and if necessary, seed recorder
            if self.checkpoint_contents is None:
                self.val_calc = ValueCalculator(lookup=lookup)
                self.recorder.update_lstg(lookup=lookup, lstg=lstg)
            else:
                self.val_calc = self.checkpoint_contents['val_calc']
                self.checkpoint_contents = None

            # simulate lstg until a stopping criterion is satisfied
            while not self.val_calc.stabilized and not time_up:
                price, T = self.simulate_lstg(environment)
                self.val_calc.add_outcome(price, T)
                if self.verbose:
                    self._print_value()
                time_up = self._is_time_up()

            # save results to value calculator
            self.recorder.add_val(self.val_calc)

            # progress bar crap
            perc = i / len(remaining_lstgs)
            if perc >= (last_perc + 0.01):
                print('Completed {}% of listings'.format(round(perc * 100)))
                last_perc += .01
                sys.stdout.flush()
            tot = (dt.now() - last_checkin).total_seconds() / 60
            if tot > 15:
                print('{} more minutes passed'.format(tot))
                last_checkin = dt.now()

            # store a checkpoint if the job is about to be killed
            if time_up:
                self.checkpoint_count += 1
                dump(self._make_checkpoint(lstg), self.checkpoint_path)
                break

        # clean up and save
        if not time_up:
            total_time = (dt.now() - self.start) / 3600
            print('TOTAL TIME: {} hours'.format(total_time))
            sys.stdout.flush()
            self.recorder.dump()
            self._delete_checkpoint()

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

    def simulate_lstg(self, environment):
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
            'checkpoint_count': self.checkpoint_count,
            'time': dt.now(),
            'val_calc': self.val_calc
        }
        return contents

    def generate_recorder(self):
        return ValueRecorder(record_path=self.records_path, verbose=self.verbose)

    @property
    def checkpoint_path(self):
        return get_checkpoint_path(self.dir, self.chunk, discrim=False)

    @property
    def records_path(self):
        return '{}{}/{}.gz'.format(self.dir, SIM_VALS_DIR, self.chunk)

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

    def _is_time_up(self):
        """
        Checks whether the generator has been running for almost 4 hours
        :return: Boolean indicating almost 4 hours has passed
        """
        tot = (dt.now() - self.start).total_seconds() / 3600
        return tot > VAL_TIME_LIMIT
