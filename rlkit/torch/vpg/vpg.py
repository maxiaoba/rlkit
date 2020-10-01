"""Vanilla Policy Gradient (REINFORCE)."""
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from rlkit.util import tensor_util as tu
from rlkit.torch.torch_rl_algorithm import TorchOnlineTrainer
from rlkit.torch.vpg.util import compute_advantages, filter_valids, pad_to_last
from rlkit.torch.optimizers import OptimizerWrapper
from rlkit.torch.pytorch_util import get_gradient_norm

class VPGTrainer(TorchOnlineTrainer):
    """Vanilla Policy Gradient (REINFORCE).

    VPG, also known as Reinforce, trains stochastic policy in an on-policy way.

    Args:
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.

    """

    def __init__(
        self,
        policy,
        value_function,
        policy_lr=1e-4,
        vf_lr=1e-3,
        policy_optimizer=None,
        vf_optimizer=None,
        vf_criterion=nn.MSELoss(),
        max_path_length=500,
        discount=0.99,
        gae_lambda=1,
        center_adv=True,
        positive_adv=False,
        policy_ent_coeff=0.0,
        use_softplus_entropy=False,
        stop_entropy_gradient=False,
        entropy_method='no_entropy',
        recurrent=False,
    ):
        super().__init__()
        self.discount = discount
        self.policy = policy
        self.max_path_length = max_path_length

        self._value_function = value_function
        self._vf_criterion = vf_criterion
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._recurrent = recurrent

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)

        if policy_optimizer is None:
            self._policy_optimizer = OptimizerWrapper(torch.optim.Adam, dict(lr=policy_lr), policy)
        else:
            self._policy_optimizer = policy_optimizer
        if vf_optimizer is None:
            self._vf_optimizer = OptimizerWrapper(torch.optim.Adam, dict(lr=vf_lr), value_function)
        else:
            self._vf_optimizer = vf_optimizer

        self._old_policy = copy.deepcopy(self.policy)

        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True

    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    def train_once(self, paths):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """
        obs, actions, rewards, returns, valids, baselines = \
            self.process_samples(paths) # num_path x T x ...

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            rewards += self._policy_ent_coeff * policy_entropies
        advs = self._compute_advantage(rewards, valids, baselines)

        if self._recurrent:
            pre_actions = actions[:,:-1,:]
            policy_input = (obs,pre_actions)
            obs_input, actions_input, rewards_input, returns_input, advs_input = \
                obs, actions, rewards, returns, advs
            valid_mask = torch.zeros(obs.shape[0],obs.shape[1]).bool()
            for i, valid in enumerate(valids):
                valid_mask[i,:valid] = True
        else:
            obs_input = torch.cat(filter_valids(obs, valids))
            actions_input = torch.cat(filter_valids(actions, valids))
            rewards_input = torch.cat(filter_valids(rewards, valids))
            returns_input = torch.cat(filter_valids(returns, valids))
            advs_input = torch.cat(filter_valids(advs, valids))
            policy_input = obs_input
            valid_mask = torch.ones(obs_input.shape[0]).bool()
            # (num of valid samples) x ...

        with torch.no_grad():
            policy_loss_before = self._compute_loss_with_adv(
                policy_input, actions_input, rewards_input, advs_input, valid_mask)
            vf_loss_before = self._compute_vf_loss(
                obs_input, returns_input, valid_mask)
            # kl_before = self._compute_kl_constraint(obs)
            kl_before = self._compute_kl_constraint(policy_input, valid_mask)

        self._train(policy_input, obs_input, actions_input, rewards_input, returns_input,
                    advs_input, valid_mask)

        with torch.no_grad():
            policy_loss_after = self._compute_loss_with_adv(
                policy_input, actions_input, rewards_input, advs_input, valid_mask)
            vf_loss_after = self._compute_vf_loss(
                obs_input, returns_input, valid_mask)
            # kl_after = self._compute_kl_constraint(obs)
            kl_after = self._compute_kl_constraint(policy_input, valid_mask)
            # policy_entropy = self._compute_policy_entropy(obs)
            policy_entropy = self._compute_policy_entropy(policy_input)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['LossBefore'] = policy_loss_before.item()
            self.eval_statistics['LossAfter'] = policy_loss_after.item()
            self.eval_statistics['dLoss'] = (policy_loss_before - policy_loss_after).item()
            self.eval_statistics['KLBefore'] = kl_before.item()
            self.eval_statistics['KL'] = kl_after.item()
            self.eval_statistics['Entropy'] = policy_entropy[valid_mask].mean().item()

            self.eval_statistics['VF LossBefore'] = vf_loss_before.item()
            self.eval_statistics['VF LossAfter'] = vf_loss_after.item()
            self.eval_statistics['VF dLoss'] = (vf_loss_before - vf_loss_after).item()

        self._old_policy = copy.deepcopy(self.policy)

    def _train(self, policy_input, obs, actions, rewards, returns, advs, valid_mask):
        r"""Train the policy and value function with minibatch.

        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.

        """
        for dataset in self._policy_optimizer.get_minibatch(
                policy_input, actions, rewards, advs, valid_mask):
            # print('_train: ',dataset[0][1][0])
            self._train_policy(*dataset)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns, valid_mask):
            self._train_value_function(*dataset)

    def _train_policy(self, obs, actions, rewards, advantages, valid_mask):
        r"""Train the policy.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated mean scalar value of policy loss (float).

        """
        self._policy_optimizer.zero_grad()
        # print('_train_policy: ',obs[1][0])
        loss = self._compute_loss_with_adv(obs, actions, rewards, advantages, valid_mask)
        loss.backward()
        self._policy_optimizer.step()

        return loss

    def _train_value_function(self, obs, returns, valid_mask):
        r"""Train the value function.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            returns (torch.Tensor): Acquired returns
                with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated mean scalar value of value function loss
                (float).

        """
        self._vf_optimizer.zero_grad()
        loss = self._compute_vf_loss(obs, returns, valid_mask)
        loss.backward()
        self._vf_optimizer.step()
        return loss

    def _compute_loss_with_adv(self, obs, actions, rewards, advantages, valid_mask):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of objective.

        """
        objectives = self._compute_objective(advantages, obs, actions, rewards)

        if self._entropy_regularzied:
            policy_entropies = self._compute_policy_entropy(obs)
            objectives += self._policy_ent_coeff * policy_entropies

        return -objectives[valid_mask].mean()

    def _compute_advantage(self, rewards, valids, baselines):
        r"""Compute mean value of loss.

        Notes: P is the maximum path length (self.max_path_length)

        Args:
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each paths
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.

        Returns:
            torch.Tensor: Calculated advantage values given rewards and
                baselines with shape :math:`(N \dot [T], )`.

        """
        advantages = compute_advantages(self.discount, self._gae_lambda,
                                        self.max_path_length, baselines,
                                        rewards)
        advantages_flat = torch.cat(filter_valids(advantages, valids))

        if self._center_adv:
            means = advantages_flat.mean()
            variance = advantages_flat.var()
            advantages = (advantages - means) / (variance + 1e-8)

        if self._positive_adv:
            advantages -= advantages.min()

        return advantages

    def _compute_kl_constraint(self, obs, valid_mask):
        r"""Compute KL divergence.

        Compute the KL divergence between the old policy distribution and
        current policy distribution.

        Notes: P is the maximum path length (self.max_path_length)

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.

        Returns:
            torch.Tensor: Calculated mean scalar value of KL divergence
                (float).

        """
        try:
            with torch.no_grad():
                old_dist = self._old_policy.get_distribution(obs)

            new_dist = self.policy.get_distribution(obs)

            kl_constraint = torch.distributions.kl.kl_divergence(
                old_dist, new_dist)

            return kl_constraint[valid_mask].mean()
        except NotImplementedError:
            return torch.tensor(0.)

    def _compute_policy_entropy(self, obs):
        r"""Compute entropy value of probability distribution.

        Notes: P is the maximum path length (self.max_path_length)

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.

        Returns:
            torch.Tensor: Calculated entropy values given observation
                with shape :math:`(N, P)`.

        """
        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = self.policy.get_distribution(obs).entropy()
        else:
            policy_entropy = self.policy.get_distribution(obs).entropy()

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_vf_loss(self, obs, returns, valid_mask):
        baselines = self._value_function(obs).squeeze(-1)
        vf_loss = self._vf_criterion(baselines[valid_mask], returns[valid_mask])
        return vf_loss

    def _compute_objective(self, advantages, obs, actions, rewards):
        r"""Compute objective value.

        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.

        """
        del rewards
        log_likelihoods = self.policy.log_prob(obs, actions)
        return log_likelihoods * advantages

    def process_samples(self, paths):
        r"""Process sample data based on the collected paths.

        Notes: P is the maximum path length (self.max_path_length)

        Args:
            paths (list[dict]): A list of collected paths

        Returns:
            torch.Tensor: The observations of the environment
                with shape :math:`(N, P, O*)`.
            torch.Tensor: The actions fed to the environment
                with shape :math:`(N, P, A*)`.
            torch.Tensor: The acquired rewards with shape :math:`(N, P)`.
            list[int]: Numbers of valid steps in each paths.
            torch.Tensor: Value function estimation at each step
                with shape :math:`(N, P)`.

        """
        valids = torch.Tensor([len(path['actions']) for path in paths]).int()
        obs = torch.stack([
            pad_to_last(path['observations'],
                        total_length=self.max_path_length,
                        axis=0) for path in paths
        ])

        actions = torch.stack([
            pad_to_last(path['actions'],
                        total_length=self.max_path_length,
                        axis=0) for path in paths
        ])

        rewards = torch.stack([
            pad_to_last(path['rewards'].reshape(-1), total_length=self.max_path_length)
            for path in paths
        ])
        returns = torch.stack([
            pad_to_last(tu.discount_cumsum(path['rewards'].reshape(-1),
                                           self.discount).copy(),
                        total_length=self.max_path_length) for path in paths
        ])

        with torch.no_grad():
            baselines = self._value_function(obs).squeeze(-1)

        return obs, actions, rewards, returns, valids, baselines

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self._value_function,
            self._old_policy,
            self.policy,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            old_policy=self._old_policy,
            value_function=self._value_function,
        )
