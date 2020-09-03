from agent.AgentComposer import AgentComposer
from agent.envs.SellerEnv import SellerEnv
from testing.agents.SellerListing import SellerListing
from testing.util import get_slr_lstgs
from testing.TestGenerator import TestGenerator


class SellerTestGenerator(TestGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose, byr=False, slr=True)

    def generate_composer(self):
        return AgentComposer(byr=False)

    def _get_valid_lstgs(self, part=None, chunk=None):
        lstgs = super()._get_valid_lstgs(part=part, chunk=chunk)
        agent_lstgs = get_slr_lstgs(chunk=chunk)
        return lstgs.intersection(agent_lstgs, sort=None)

    def simulate_lstg(self):
        params = self._get_listing_params()
        lstg_log = SellerListing(params=params)
        self.query_strategy.update_log(lstg_log)
        obs = self.env.reset(next_lstg=False)
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
            self.simulate_lstg()

    @property
    def env_class(self):
        return SellerEnv
