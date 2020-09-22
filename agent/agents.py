import torch
from torch.distributions import Beta
from rlpyt.agents.base import AgentStep
from rlpyt.agents.pg.base import AgentInfo
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.distributions.categorical import DistInfo
from rlpyt.utils.buffer import buffer_to
from torch.nn.functional import softmax, softplus
from constants import EPS


def parse_value_params(value_params):
    p = softmax(value_params[:, :-2], dim=-1)
    beta_params = softplus(torch.clamp(value_params[:, -2:], min=-5))
    a, b = beta_params[:, 0], beta_params[:, 1]
    return p, a, b


class SplitCategoricalPgAgent(CategoricalPgAgent):
    def __init__(self, serial=False, **kwargs):
        self.serial = serial
        super().__init__(**kwargs)

    def __call__(self, observation, prev_action, prev_reward):
        model_inputs = self._model_inputs(observation, prev_action, prev_reward)
        pdf, value_params = self.model(*model_inputs)
        if not self.serial:
            pdf = pdf.squeeze()
            value_params = value_params.squeeze()
        return buffer_to((DistInfo(prob=pdf), value_params), device="cpu")

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = self._model_inputs(observation, prev_action, prev_reward)
        pdf, value_params = self.model(*model_inputs)
        v = self._calculate_value(value_params)
        if not self.serial:
            pdf = pdf.squeeze()
            v = v.squeeze()
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
        if not self.serial:
            v = v.squeeze()
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
        raise NotImplementedError()

    @staticmethod
    def get_value_loss(value_params, return_, valid):
        raise NotImplementedError()


class SellerAgent(SplitCategoricalPgAgent):

    def _calculate_value(self, value_params):
        p, a, b = parse_value_params(value_params)
        beta_mean = a / (a + b)
        v = p[:, 1] + p[:, -1] * beta_mean
        return v

    @staticmethod
    def get_value_loss(value_params, return_, valid):
        p, a, b = parse_value_params(value_params)
        idx0 = torch.isclose(return_, torch.zeros_like(return_)) & valid  # no sale
        lnL = torch.sum(torch.log(p[idx0, 0] + EPS))
        idx1 = torch.isclose(return_, torch.ones_like(return_)) & valid  # sells for list price
        lnL += torch.sum(torch.log(p[idx1, 1] + EPS))
        idx_beta = ~idx0 & ~idx1 & valid  # intermediate outcome
        lnL += torch.sum(torch.log(p[idx_beta, -1] + EPS))
        lnL += torch.sum(Beta(a[idx_beta], b[idx_beta]).log_prob(return_[idx_beta]))
        return -lnL


class BuyerMarketAgent(SplitCategoricalPgAgent):

    def _calculate_value(self, value_params):
        p, a, b = parse_value_params(value_params)
        beta_mean = a / (a + b) * 2 - 1  # on [-1, 1]
        v = p[:, -1] * beta_mean
        return v

    @staticmethod
    def get_value_loss(value_params, return_, valid):
        p, a, b = parse_value_params(value_params)
        idx0 = torch.isclose(return_, torch.zeros_like(return_)) & valid  # no sale
        lnL = torch.sum(torch.log(p[idx0, 0] + EPS))
        idx_beta = ~idx0 & valid
        lnL += torch.sum(torch.log(p[idx_beta, -1] + EPS))
        return_beta = torch.clamp((return_[idx_beta] + 1) / 2,
                                  min=EPS, max=1 - EPS)
        dist = Beta(a[idx_beta], b[idx_beta])
        lnL += torch.sum(dist.log_prob(return_beta))
        return -lnL


class BuyerBINAgent(SplitCategoricalPgAgent):

    def _calculate_value(self, value_params):
        p, a, b = parse_value_params(value_params)
        v = p[:, 1] + p[:, -1] * a / (a + b)
        return v

    @staticmethod
    def get_value_loss(value_params, return_, valid):
        p, a, b = parse_value_params(value_params)
        idx0 = torch.isclose(return_, torch.zeros_like(return_)) & valid  # no sale
        lnL = torch.sum(torch.log(p[idx0, 0] + EPS))
        idx1 = torch.isclose(return_, torch.ones_like(return_)) & valid
        lnL += torch.sum(torch.log(p[idx1, 1] + EPS))
        idx_beta = ~idx0 & ~idx1 & valid  # intermediate outcome
        lnL += torch.sum(torch.log(p[idx_beta, -1] + EPS))
        dist = Beta(a[idx_beta], b[idx_beta])
        lnL += torch.sum(dist.log_prob(return_[idx_beta]))
        return -lnL
