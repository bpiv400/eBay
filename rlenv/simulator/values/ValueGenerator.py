from statistics import mean, variance
from rlenv.env_consts import VAL_SE_CHECK, MIN_SALES, SILENT
from rlenv.env_utils import get_checkpoint_path, get_chunk_dir
from rlenv.simulator.Generator import Generator
from rlenv.simulator.values.ValueCalculator import ValueCalculator
from rlenv.simulator.values.ValueRecorder import ValueRecorder


class ValueGenerator(Generator):
    def __init__(self, direct=None, num=None):
        super(ValueGenerator, self).__init__(direct=direct, num=num)
        # noinspection PyTypeChecker
        self.val_calc = None  # type: ValueCalculator
        self.recorder = self.make_recorder()

    def setup_env(self, lstg, lookup):
        """
        Invokes Generator.setup_env to create environment and
        initializes a value calculator for the current lstg
        """
        if self.checkpoint_contents is not None:
            # grab the value calculator and reset checkpoint
            self.val_calc = self.checkpoint_contents['val_calc']
        else:
            self.val_calc = ValueCalculator(lookup=lookup)
        env = super(ValueGenerator, self).setup_env(lstg, lookup)
        # if there's a checkpoint and this is the first lstg
        return env

    def simulate_lstg_loop(self, environment):
        """
        Simulates a particular listing until the value estimate se is beneath the given tolerance
        :param environment: RewardEnvironment
        :return: Boolean indicating whether the job has run out of queue time
        """
        stop, time_up = False, False
        while not stop and not time_up:
            (sale, price, dur), time_up = self.simulate_lstg(environment)
            self.val_calc.add_outcome(sale, price)
            stop = self._update_stop()
            self._print_value(stop)
            if self.val_calc.exp_count % 100 == 0:
                self.mem_check()
        return time_up

    def _print_value(self, stop):
        """
        Prints information about the most recent value calculation
        """
        if self.val_calc.exp_count % VAL_SE_CHECK == 0 and not SILENT:
            print('Trial: {}'.format(self.val_calc.exp_count))
            if len(self.val_calc.sales) == 0:
                print('No Sales')
            elif len(self.val_calc.sales) < MIN_SALES:
                print('fewer than min sales ({} / {})'.format(len(self.val_calc.sales), MIN_SALES))
            else:
                print('cut: {} | tol: {}'.format(self.val_calc.cut, self.val_calc.se_tol))
                print('rate of no sale: {}'.format(self.val_calc.p))
                price_header = 'sale count: {} | price mean: {}'.format(len(self.val_calc.sales),
                                                                        mean(self.val_calc.sales))
                price_header = '{} | price var: {}'.format(price_header, variance(self.val_calc.sales))
                print(price_header)
                print('mean of value: {} | mean standard error: {}'.format(self.val_calc.mean,
                                                                           self.val_calc.mean_se))
                if stop:
                    print('Estimation stabilized')
                else:
                    print('Predicted trials remaining: {}'.format(self.val_calc.trials_until_stable))

    def make_checkpoint(self, lstg):
        contents = super(ValueGenerator, self).make_checkpoint(lstg)
        contents['val_calc'] = self.val_calc
        return contents

    def make_recorder(self):
        return ValueRecorder(self.records_path)

    def _update_stop(self):
        """
        Checks whether a the generator may stop simulating a particular lstg
        based on whether the value estimate has stabilized
        :return: Boolean indicating whether to stop
        """
        counter = self.val_calc.exp_count
        if self.val_calc.has_sales and counter % VAL_SE_CHECK == 0:
            return self.val_calc.stabilized
        else:
            return False

    @property
    def checkpoint_path(self):
        return get_checkpoint_path(self.dir, self.chunk, discrim=False)

    @property
    def records_path(self):
        return get_chunk_dir(self.dir, self.chunk, discrim=False)