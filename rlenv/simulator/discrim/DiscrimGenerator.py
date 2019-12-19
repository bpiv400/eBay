from rlenv.env_consts import SIM_COUNT
from rlenv.env_utils import get_checkpoint_path, get_chunk_dir
from rlenv.simulator.Generator import Generator
from rlenv.simulator.discrim.DiscrimRecorder import DiscrimRecorder


class DiscrimGenerator(Generator):
    def __init__(self, direct=None, num=None):
        super(DiscrimGenerator, self).__init__(direct=direct, num=num)
        self.recorder = self.make_recorder()

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
        self.mem_check()
        return time_up

    def make_recorder(self):
        return DiscrimRecorder(self.records_path)

    @property
    def checkpoint_path(self):
        return get_checkpoint_path(self.dir, self.chunk, discrim=True)

    @property
    def records_path(self):
        return get_chunk_dir(self.dir, self.chunk, discrim=True)
