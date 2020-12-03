from agent.AgentComposer import AgentComposer
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from testing.agents.BuyerListing import BuyerListing
from testing.agents.SellerListing import SellerListing
from testing.TestGenerator import TestGenerator
from featnames import BYR_HIST


class AgentTestGenerator(TestGenerator):

    def simulate_lstg(self):
        # create listing log
        params = self._get_listing_params()
        if self.byr:
            params['thread_id'] = 1
            lstg_log = BuyerListing(params=params)
        else:
            lstg_log = SellerListing(params=params)
        self.query_strategy.update_log(lstg_log)

        # initialize environment
        hist = None if not self.byr else self.loader.x_thread.loc[1, BYR_HIST]
        obs = self.env.reset(hist=hist)
        agent_tuple = obs, None, None, None

        # run environment to end
        if obs is not None:
            done = False
            while not done:
                action = lstg_log.get_action(agent_tuple=agent_tuple)
                agent_tuple = self.env.step(action)
                done = agent_tuple is None or agent_tuple[2]

        lstg_log.verify_done()

    def generate_composer(self):
        return AgentComposer(byr=self.byr)

    @property
    def env_class(self):
        return BuyerEnv if self.byr else SellerEnv
