from collections import OrderedDict
import numpy as np
import torch
from rlpyt.agents.base import AgentStep
from rlpyt.agents.pg.base import AgentInfo
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.distributions.categorical import DistInfo
from rlpyt.utils.buffer import buffer_to
from agent.agents.util import valid_from_done, backward_from_done
from agent.const import AGENT_CONS


class EBayAgent(CategoricalPgAgent):
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
        opt_info['ActionsPerTraj'] = num_actions
        opt_info['DaysToDone'] = info.days[done].numpy()

        # action stats
        action = samples.agent.action[valid].numpy()
        turn = env.env_info.turn[valid].numpy()
        sets = np.stack([AGENT_CONS[t] for t in turn])
        con = np.take_along_axis(sets, np.expand_dims(action, 1), 1)

        opt_info = self._get_turn_info(opt_info=opt_info,
                                       env=env,
                                       turn=turn,
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

    def _get_turn_info(self, opt_info=None, env=None, turn=None,
                       con=None, valid=None):
        raise NotImplementedError()