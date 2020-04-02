from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging import logger

class EbayRunner(MinibatchRl):
    def __init__(self,
                 algo,
                 agent,
                 sampler,
                 seed=None,
                 affinity=None,
                 batch_size=None):
        super().__init__(algo=algo, agent=agent, sampler=sampler,
                         seed=seed, affinity=affinity, n_steps=batch_size,
                         log_interval_steps=batch_size)
        self.itr_ = 0


    def train(self):
        if self.itr_ == 0:
            self.startup()

