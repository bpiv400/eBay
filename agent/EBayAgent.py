import torch
from rlpyt.agents.base import AgentStep, BaseAgent
from rlpyt.agents.pg.base import AgentInfo
from rlpyt.utils.buffer import buffer_to
from agent.BetaCategorical import BetaCategorical
from agent.util import pack_dist_info


class EBayAgent(BaseAgent):
    """
    Agent for policy gradient algorithm using beta-categorical action distribution.
    """

    def __call__(self, observation, prev_action=None, prev_reward=None):
        """Performs forward pass on training data, for algorithm."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        params, value = self.model(*model_inputs)
        dist_info = pack_dist_info(params)
        return buffer_to((dist_info, value), device="cpu")

    def initialize(self, env_spaces, share_memory=False,
                   global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
                           global_B=global_B, env_ranks=env_ranks)
        self.distribution = BetaCategorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, and value estimate.
        Moves inputs to device and returns outputs back to CPU, for the
        sampler.  (no grad)
        """
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        params, value = self.model(*model_inputs)
        dist_info = pack_dist_info(params)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        """
        Compute the value estimate for the environment state, e.g. for the
        bootstrap value, V(s_{T+1}), in the sampler.  (no grad)
        """
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        _, value = self.model(*model_inputs)
        return value.to("cpu")

    def value_parameters(self):
        return self.model.value_net.parameters()

    def policy_parameters(self):
        return self.model.policy_net.parameters()
