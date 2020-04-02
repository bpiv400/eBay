from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging import logger


class EbayRunner(MinibatchRl):
    def __init__(self,
                 algo,
                 agent,
                 sampler,
                 seed=None,
                 affinity=None,
                 batches_per_evaluation=None,
                 batch_size=None):
        # initialized in super class
        self.sampler = None
        self.agent = None
        self.algo = None
        super().__init__(algo=algo, agent=agent, sampler=sampler,
                         seed=seed, affinity=affinity,
                         n_steps=batch_size * batches_per_evaluation,
                         log_interval_steps=batch_size)
        self.batches_per_evaluation = batches_per_evaluation
        self.itr_ = 0

    def update_agent(self, agent):
        self.agent = agent

    def update_algo(self, algo):
        self.algo = algo

    def train(self):
        n_itr = self.startup()
        assert n_itr == self.batches_per_evaluation
        for itr in range(n_itr):
            logger.set_iteration(itr + self.itr_)
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    self.log_diagnostics(itr)

        self.itr_ += self.batches_per_evaluation
        self.shutdown()
