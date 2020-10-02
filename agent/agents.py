from collections import OrderedDict
import numpy as np
import torch
from torch.distributions import Beta
from rlpyt.agents.base import AgentStep
from rlpyt.agents.pg.base import AgentInfo
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.distributions.categorical import DistInfo
from rlpyt.utils.buffer import buffer_to
from torch.nn.functional import softmax, softplus
from agent.util import backward_from_done, valid_from_done
from agent.const import BYR_MIN_CON1
from constants import EPS, IDX, SLR
from featnames import BYR


def parse_value_params(value_params):
    p = softmax(value_params[:, :-2], dim=-1)
    beta_params = softplus(torch.clamp(value_params[:, -2:], min=-5))
    a, b = beta_params[:, 0], beta_params[:, 1]
    return p, a, b


class SplitCategoricalPgAgent(CategoricalPgAgent):
    def __init__(self, serial=False, **kwargs):
        super().__init__(**kwargs)
        self.serial = serial

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

    def process_samples(self, samples=None):
        opt_info = OrderedDict()

        # break out samples
        env = samples.env
        reward, done, info = (env.reward, env.done, env.env_info)

        # ignore steps from unfinished trajectories
        valid = valid_from_done(done)

        # various counts
        num_actions = info.num_actions[done].numpy()
        num_threads = info.num_threads[done].numpy()
        opt_info['ActionsPerTraj'] = num_actions
        opt_info['ThreadsPerTraj'] = num_threads
        opt_info['DaysToDone'] = info.days[done].numpy()

        # action stats
        action = samples.agent.action[valid].numpy()
        con = np.take_along_axis(self.model.con_space, action, 0)

        opt_info = self._get_turn_info(opt_info=opt_info,
                                       env=env,
                                       con=con,
                                       valid=valid)

        # reward stats
        traj_value = reward[done]
        opt_info['DollarReturn'] = traj_value.numpy()
        opt_info['NormReturn'] = (traj_value / info.max_return[done]).numpy()
        opt_info['Rate_Sale'] = np.mean(info.agent_sale[done].numpy())

        # propagate return from end of trajectory and normalize
        return_ = backward_from_done(x=reward, done=done) / info.max_return

        return opt_info, valid, return_

    def _get_turn_info(self, opt_info=None, env=None, con=None, valid=None):
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
        zeros = torch.zeros_like(return_)

        # no sale (and worth zero)
        idx0 = torch.isclose(return_, zeros) & valid
        lnL = torch.sum(torch.log(p[idx0, 0] + EPS))

        # sells for list price
        idx1 = torch.isclose(return_, zeros + 1) & valid
        lnL += torch.sum(torch.log(p[idx1, 1] + EPS))

        # intermediate outcome
        idx_beta = ~idx0 & ~idx1 & valid
        lnL += torch.sum(torch.log(p[idx_beta, -1] + EPS))
        dist = Beta(a[idx_beta], b[idx_beta])
        lnL += torch.sum(dist.log_prob(return_[idx_beta]))

        return -lnL

    def _get_turn_info(self, opt_info=None, env=None, con=None, valid=None):
        turn = env.env_info.turn[valid].numpy()
        for t in IDX[SLR]:
            # rate of reaching turn t
            opt_info['Rate_{}'.format(t)] = np.mean(turn == t)

            # accept, reject, and expiration rates
            con_t = con[turn == t]
            prefix = 'Turn{}'.format(t)
            opt_info['{}_{}'.format(prefix, 'AccRate')] = np.mean(con_t == 1)
            opt_info['{}_{}'.format(prefix, 'ExpRate')] = np.mean(con_t > 1)
            opt_info['{}_{}'.format(prefix, 'RejRate')] = np.mean(con_t == 0)

            # moments of concession distribution
            opt_info['{}_{}'.format(prefix, 'ConRate')] = \
                np.mean((con_t > 0) & (con_t < 1))
            opt_info['{}{}'.format(prefix, 'Con')] = \
                con_t[(con_t > 0) & (con_t < 1)]
        return opt_info


class BuyerAgent(SplitCategoricalPgAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.length = 2 - BYR_MIN_CON1  # of [-1, 1 - BYR_MIN_CON1]
        self.max_return = self.length - 1

    def _calculate_value(self, value_params):
        p, a, b = parse_value_params(value_params)
        beta_mean = a / (a + b) * self.length - 1
        v = p[:, 2] * self.max_return - p[:, 1] + p[:, -1] * beta_mean
        return v

    def get_value_loss(self, value_params, return_, valid):
        p, a, b = parse_value_params(value_params)
        zeros = torch.zeros_like(return_)

        # no sale
        idx0 = torch.isclose(return_, zeros) & valid
        lnL = torch.sum(torch.log(p[idx0, 0] + EPS))

        # worth 0, sells for list price
        idx1 = torch.isclose(return_, zeros - 1) & valid
        lnL += torch.sum(torch.log(p[idx1, 1] + EPS))

        # worth 1, sells for minimum offer
        idx2 = torch.isclose(return_, zeros + self.max_return) & valid
        lnL += torch.sum(torch.log(p[idx2, 2] + EPS))

        # intermediate outcome
        idx_beta = ~idx0 & ~idx1 & ~idx2 & valid
        lnL += torch.sum(torch.log(p[idx_beta, -1] + EPS))
        dist = Beta(a[idx_beta], b[idx_beta])
        norm_return = (return_[idx_beta] + 1) / self.length
        lnL += torch.sum(dist.log_prob(norm_return))

        return -lnL

    def _get_turn_info(self, opt_info=None, env=None, con=None, valid=None):
        turn = env.env_info.turn[valid].numpy()
        turn_ct = backward_from_done(x=env.env_info.turn, done=env.done)
        turn_ct = turn_ct[valid].numpy()
        for t in IDX[BYR]:
            # rate of reaching turn t
            opt_info['Rate_{}'.format(t)] = np.mean(turn_ct == t)

            # accept and reject rates and moments of concession distribution
            con_t = con[turn == t]
            prefix = 'Turn{}'.format(t)
            opt_info['{}_{}'.format(prefix, 'AccRate')] = np.mean(con_t == 1)
            if t < 7:
                opt_info['{}_{}'.format(prefix, 'RejRate')] = np.mean(con_t == 0)
                opt_info['{}_{}'.format(prefix, 'ConRate')] = \
                    np.mean((con_t > 0) & (con_t < 1))
                opt_info['{}{}'.format(prefix, 'Con')] = \
                    con_t[(con_t > 0) & (con_t < 1)]
        return opt_info
