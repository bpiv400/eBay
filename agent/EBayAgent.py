from collections import OrderedDict
import numpy as np
import torch
from rlpyt.agents.base import AgentStep
from rlpyt.agents.pg.base import AgentInfo
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.distributions.categorical import DistInfo
from rlpyt.utils.buffer import buffer_to
from utils import get_role
from agent.const import AGENT_CONS
from constants import IDX, SLR


class EBayAgent(CategoricalPgAgent):
    def __init__(self, serial=False, byr=False, **kwargs):
        super().__init__(**kwargs)
        self.serial = serial
        self.byr = byr

    def __call__(self, observation, prev_action=None, prev_reward=None):
        x = buffer_to(observation, device=self.device)
        pdf, v = self.model(x)
        if not self.serial:
            pdf = pdf.squeeze()
            v = v.squeeze()
        return buffer_to((DistInfo(prob=pdf), v), device="cpu")

    @torch.no_grad()
    def step(self, observation, prev_action=None, prev_reward=None):
        x = buffer_to(observation, device=self.device)
        pdf, v = self.model(x)
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
        x = buffer_to(observation, device=self.device)
        v = self.model(x, value_only=True)
        if not self.serial:
            v = v.squeeze()
        return v.to("cpu")

    def value_parameters(self):
        return self.model.value_net.parameters()

    def policy_parameters(self):
        return self.model.policy_net.parameters()

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
        last_turn = env.env_info.turn[env.done].numpy()

        opt_info = self._add_turn_info(
            turn=turn, last_turn=last_turn, opt_info=opt_info, con=con)

        # reward stats
        traj_value = reward[done]
        opt_info['DollarReturn'] = traj_value.numpy()
        opt_info['NormReturn'] = (traj_value / info.max_return[done]).numpy()
        opt_info['Rate_Sale'] = np.mean(info.agent_sale[done].numpy())

        # propagate return from end of trajectory and normalize
        return_ = backward_from_done(x=reward, done=done) / info.max_return

        return opt_info, valid, return_

    def _add_turn_info(self, turn=None, last_turn=None, opt_info=None, con=None):
        for t in IDX[get_role(self.byr)]:
            # rate of reaching turn t
            opt_info['Rate_{}'.format(t)] = np.mean(last_turn == t)

            # accept and reject rates and moments of concession distribution
            con_t = con[turn == t]
            prefix = 'Turn{}'.format(t)
            opt_info['{}_{}'.format(prefix, 'AccRate')] = np.mean(con_t == 1)
            if t in IDX[SLR]:
                opt_info['{}_{}'.format(prefix, 'ExpRate')] = np.mean(con_t > 1)
            if t < 7:
                opt_info['{}_{}'.format(prefix, 'RejRate')] = np.mean(con_t == 0)
                opt_info['{}_{}'.format(prefix, 'ConRate')] = \
                    np.mean((con_t > 0) & (con_t < 1))
                opt_info['{}{}'.format(prefix, 'Con')] = \
                    con_t[(con_t > 0) & (con_t < 1)]
        return opt_info


def backward_from_done(x=None, done=None):
    """
    Propagates value at done across trajectory. Operations
    vectorized across all trailing dimensions after the first [T,].
    :param tensor x: tensor to propagate across trajectory
    :param tensor done: indicator for end of trajectory
    :return tensor newx: value at done at every step of trajectory
    """
    dtype = x.dtype  # cast new tensors to this data type
    num_t, num_envs = x.shape  # time steps, number of envs

    # recast
    done = done.type(torch.int)

    # initialize output tensor
    newx = torch.zeros(x.shape, dtype=dtype)

    # vector for given time period
    v = torch.zeros(num_envs, dtype=dtype)

    for t in reversed(range(num_t)):
        v = v * (1 - done[t]) + x[t] * done[t]
        newx[t] += v

    return newx


def valid_from_done(done):
    """Returns a float mask which is zero for all time-steps after the last
    `done=True` is signaled.  This function operates on the leading dimension
    of `done`, assumed to correspond to time [T,...], other dimensions are
    preserved."""
    done = done.type(torch.float)
    done_count = torch.cumsum(done, dim=0)
    done_max, _ = torch.max(done_count, dim=0)
    valid = torch.abs(done_count - done_max) + done
    valid = torch.clamp(valid, max=1)
    return valid.bool()
