from env_consts import SIM_COUNT
from env_utils import get_checkpoint_path, get_chunk_dir
from simulator.Generator import Generator


class DiscrimGenerator(Generator):
    def __init__(self, direct=None, num=None):
        super(DiscrimGenerator, self).__init__(direct=direct, num=num)

    def simulate_lstg_loop(self, environment):
        """
        Simulates a particular listing a given number of times and stores
        outputs required to train discrimator
        :param environment: RewardEnvironment
        :return: Boolean indicating whether the job has run out of queue time
        """
        time_up = False
        while self.recorder.sim < SIM_COUNT - 1 and not time_up:
            _, time_up = self.simulate_lstg(environment)
        return time_up

    @property
    def checkpoint_path(self):
        return get_checkpoint_path(self.dir, self.chunk, discrim=True)

    @property
    def records_path(self):
        return get_chunk_dir(self.dir, self.chunk, discrim=True)
