import torch
from rlpyt.agents.base import AgentStep
from rlpyt.agents.pg.base import AgentInfo
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.distributions.categorical import DistInfo
from rlpyt.utils.buffer import buffer_to


class SplitCategoricalPgAgent(CategoricalPgAgent):
    def __init__(self, byr=False, serial=False, **kwargs):
        self.byr = byr
        self.serial = serial
        super().__init__(**kwargs)

    def __call__(self, observation, prev_action, prev_reward):
        model_inputs = self._model_inputs(observation, prev_action, prev_reward)
        pdf, value_params = self.model(*model_inputs)
        if not self.serial:
            pdf = pdf.squeeze()
            value_params = tuple(elem.squeeze() for elem in value_params)
        return buffer_to((DistInfo(prob=pdf), value_params), device="cpu")

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = self._model_inputs(observation, prev_action, prev_reward)
        pdf, value_params = self.model(*model_inputs)
        v = self._calculate_value(value_params)
        if not self.serial:
            pdf = pdf.squeeze()
        dist_info = DistInfo(prob=pdf)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=v)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        model_inputs = self._model_inputs(observation, prev_action, prev_reward)
        value_params = self.model(*model_inputs, value_only=True)
        v = self._calculate_value(value_params)
        return v.to("cpu")

    def value_parameters(self):
        return self.model.value_net.parameters()

    def policy_parameters(self):
        return self.model.policy_net.parameters()

    def _model_inputs(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        return model_inputs

    def _calculate_value(self, value_params):
        p, a, b = value_params
        if self.byr:
            v = p[:, -1] * (a / (a + b) * 2 - 1)
        else:
            v = p[:, 1] + p[:, -1] * a / (a + b)
        if not self.serial:
            v = v.squeeze()
        return v
