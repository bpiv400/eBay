from agent.AgentComposer import AgentComposer
from agent.envs.BuyerEnv import BuyerEnv
from agent.envs.SellerEnv import SellerEnv
from testing.agents.BuyerListing import BuyerListing
from testing.agents.SellerListing import SellerListing
from testing.TestGenerator import TestGenerator
from testing.util import get_agent_lstgs


class AgentTestGenerator(TestGenerator):
    def __init__(self, verbose=False, byr=False, agent_thread=None, slr=False):
        self.agent_thread = agent_thread
        super().__init__(verbose=verbose, byr=byr, slr=slr)

    def simulate_lstg(self):
        # create listing log
        params = self._get_listing_params()
        if self.byr:
            params['thread_id'] = self.agent_thread
            lstg_log = BuyerListing(params=params)
        else:
            lstg_log = SellerListing(params=params)
        self.query_strategy.update_log(lstg_log)

        # initialize environment
        obs = self.env.reset()
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

    def generate_env(self):
        if self.byr:
            env_args = self.env_args.copy()
            env_args['agent_thread'] = self.agent_thread
            return self.env_class(**env_args)
        else:
            return super().generate_env()

    def _get_valid_lstgs(self, part=None, chunk=None):
        """
        Retrieves a list of lstgs from the chunk where the agents makes at least one
        turn.

        Verifies this list matches exactly the lstgs with inputs for the relevant model
        :return: pd.Int64Index
        """
        lstgs = super()._get_valid_lstgs(part=part, chunk=chunk)
        agent_lstgs = get_agent_lstgs(chunk=chunk, byr=self.byr)
        lstgs = lstgs.intersection(agent_lstgs, sort=None)
        return lstgs
