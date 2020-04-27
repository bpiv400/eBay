import torch
from rlpyt.algos.pg.ppo import PPO
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel

EPS = 1e-8


def kl(p, q):
    return torch.sum(p * (torch.log(p + EPS) - torch.log(q + EPS)), dim=-1)


def mean_kl(p, q, valid=None):
    return valid_mean(kl(p, q), valid)


class CrossEntropyPPO(PPO):
    """
    Swaps entropy bonus with cross-entropy penalty, where cross-entropy
    is calculated using the policy from the initialized agent.
    """

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False
    ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

        # parameters to be definied later
        self._batch_size = None
        self._ratio_clip = None
        self.lr_scheduler = None
        self.init_agent = None

    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        # init_agent new initialized agent
        self.init_agent = PgCategoricalAgentModel(
            **self.agent.model_kwargs).to('cuda:0')
        # original PPO initialization
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info,
             init_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            raise NotImplementedError()

        # agent's policy
        dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        # loss from policy
        ratio = dist.likelihood_ratio(action,
                                      old_dist_info=old_dist_info,
                                      new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
                                    1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        # loss from value estimation
        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        # cross-entropy loss
        init_dist_info, _ = self.init_agent(*agent_inputs)
        cross_entropy = mean_kl(init_dist_info.to('cpu'),
                                dist_info.prob,
                                valid)
        cross_entropy_loss = self.entropy_loss_coeff * cross_entropy

        # total loss
        loss = pi_loss + value_loss + cross_entropy_loss

        # perplexity
        perplexity = dist.mean_perplexity(dist_info, valid)

        # cross-entropy replaces entropy
        return loss, cross_entropy, perplexity
