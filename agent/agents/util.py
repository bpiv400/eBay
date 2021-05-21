import torch
from torch.nn.functional import softmax, softplus


def parse_value_params(value_params):
    p = softmax(value_params[:, :-2], dim=-1)
    beta_params = softplus(torch.clamp(value_params[:, -2:], min=-5))
    a, b = beta_params[:, 0], beta_params[:, 1]
    return p, a, b


def backward_from_done(x=None, done=None):
    """
    Propagates value at done across trajectory. Operations
    vectorized across all trailing dimensions after the first [T,].
    :param tensor x: tensor to propagate across trajectory
    :param tensor done: indicator for end of trajectory
    :return tensor newx: value at done at every step of trajectory
    """
    dtype = x.dtype  # cast new tensors to this data type
    T, N = x.shape  # time steps, number of envs

    # recast
    done = done.type(torch.int)

    # initialize output tensor
    newx = torch.zeros(x.shape, dtype=dtype)

    # vector for given time period
    v = torch.zeros(N, dtype=dtype)

    for t in reversed(range(T)):
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
