from agent.AgentComposer import AgentComposer
from agent.envs.BuyerEnv import BuyerEnv
from testing.agents.BuyerListing import BuyerListing
from testing.util import get_byr_lstgs
from testing.TestGenerator import TestGenerator
from featnames import THREAD, BYR_HIST


class BuyerTestGenerator(TestGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose, byr=True, slr=False)

    def generate_composer(self):
        return AgentComposer(byr=True)

    def _get_valid_lstgs(self, part=None, chunk=None):
        lstgs = super()._get_valid_lstgs(part=part, chunk=chunk)
        agent_lstgs = get_byr_lstgs(chunk=chunk)
        return lstgs.intersection(agent_lstgs, sort=None)

    def _count_rl_buyers(self):
        # there should always be at least 1 if initial agents subsetting
        # was correct
        threads = self.loader.x_offer.index.get_level_values(THREAD)
        return len(threads.unique())

    def simulate_lstg(self, thread_id=None):
        params = self._get_listing_params()
        params['thread_id'] = thread_id
        lstg_log = BuyerListing(params=params)
        self.query_strategy.update_log(lstg_log)
        hist = self.loader.x_thread.loc[thread_id, BYR_HIST]
        obs = self.env.reset(next_lstg=False, hist=hist)
        agent_tuple = obs, None, None, None
        if obs is not None:
            done = False
            while not done:
                action = lstg_log.get_action(agent_tuple=agent_tuple)
                agent_tuple = self.env.step(action)
                done = agent_tuple is None or agent_tuple[2]
        lstg_log.verify_done()

    def generate(self):
        while self.env.has_next_lstg():
            self.env.next_lstg()
            num_buyers = self._count_rl_buyers()
            for thread_id in range(1, num_buyers + 1):
                # simulate lstg once for each buyer
                if self.verbose:
                    print('\nAgent is thread {}'.format(thread_id))
                self.simulate_lstg(thread_id=thread_id)

    @property
    def env_class(self):
        return BuyerEnv
