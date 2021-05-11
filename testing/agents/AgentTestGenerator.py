from agent.AgentComposer import AgentComposer
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from testing.Listing import Listing
from testing.agents.BuyerListing import BuyerListing
from testing.agents.SellerListing import SellerListing
from testing.TestGenerator import TestGenerator
from testing.util import get_slr_lstgs


class AgentTestGenerator(TestGenerator):
    def __init__(self, verbose=False, byr=False, slr=False):
        super().__init__(verbose=verbose, byr=byr, slr=slr)

    def simulate_lstg(self):
        # create listing log
        params = self._get_listing_params()
        lstg_class = Listing if self.byr else SellerListing
        lstg_log = lstg_class(params=params)
        self.query_strategy.update_log(lstg_log)

        # initialize environment
        obs = self.env.reset()
        agent_tuple = obs, None, None, None

        # after agent_thread is drawn, use BuyerListing
        if self.byr:
            params['thread_id'] = self.env.agent_thread
            lstg_log = BuyerListing(params=params)
            self.query_strategy.update_log(lstg_log)

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

    def _get_valid_lstgs(self, part=None, chunk=None):
        """
        Retrieves a list of lstgs from the chunk where the agents makes at least one
        turn.

        Verifies this list matches exactly the lstgs with inputs for the relevant model
        :return: pd.Int64Index
        """
        lstgs = super()._get_valid_lstgs(part=part, chunk=chunk)
        if not self.byr:
            agent_lstgs = get_slr_lstgs(chunk=chunk)
            lstgs = lstgs.intersection(agent_lstgs, sort=None)
        return lstgs
