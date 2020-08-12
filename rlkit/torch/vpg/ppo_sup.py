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
from rlkit.torch.core import torch_ify
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict

class PPOSupTrainer(TorchOnlineTrainer):
    """PPO + supervised learning.

    Args:
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        max_episode_length (int): Maximum length of a single rollout.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies.
        num_train_per_epoch (int): Number of train_once calls per epoch.
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

    def __init__(self,
                 policy,
                 value_function,
                 sup_learners,
                 replay_buffer,
                 exploration_bonus,
                 policy_lr=2.5e-4,
                 vf_lr=2.5e-4,
                 vf_criterion=nn.MSELoss(),
                 sup_lr=1e-3,
                 sup_batch_size=64,
                 sup_train_num=1,
                 max_path_length=500,
                 lr_clip_range=2e-1,
                 discount=0.99,
                 gae_lambda=0.97,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
    ):
        super().__init__()
        self.discount = discount
        self.policy = policy
        self.sup_learners = sup_learners
        self.replay_buffer = replay_buffer
        self.sup_batch_size = sup_batch_size
        self.sup_train_num = sup_train_num
        self.max_path_length = max_path_length
        self.exploration_bonus = exploration_bonus

        self._value_function = value_function
        self._vf_criterion = vf_criterion
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._lr_clip_range = lr_clip_range

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)

        self._policy_optimizer = OptimizerWrapper(
            torch.optim.Adam, 
            dict(lr=policy_lr),
            policy,
            max_optimization_epochs=10,
            minibatch_size=64)

        self._vf_optimizer = OptimizerWrapper(
            torch.optim.Adam,
            dict(lr=vf_lr),
            value_function,
            max_optimization_epochs=10,
            minibatch_size=64)

        self._sup_optimizers = []
        for sup_learner in self.sup_learners:
            self._sup_optimizers.append(
                torch.optim.Adam(sup_learner.parameters(),
                                lr=sup_lr)
                )

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
        obs, actions, rewards, returns, valids, baselines, n_labels = \
            self.process_samples(paths)

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = torch.cat(filter_valids(obs, valids))
        actions_flat = torch.cat(filter_valids(actions, valids))
        rewards_flat = torch.cat(filter_valids(rewards, valids))
        returns_flat = torch.cat(filter_valids(returns, valids))
        advs_flat = self._compute_advantage(rewards, valids, baselines)
        n_labels_flat = [torch.cat(filter_valids(labels, valids)) for labels in n_labels]

        self.replay_buffer.add_batch(obs_flat, n_labels_flat)
        for _ in range(self.sup_train_num):
            batch = self.replay_buffer.random_batch(self.sup_batch_size)
            sup_losses = self._train_sup_learners(batch['observations'],batch['n_labels'])

        with torch.no_grad():
            policy_loss_before = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_before = self._compute_vf_loss(
                obs_flat, returns_flat)
            # kl_before = self._compute_kl_constraint(obs)
            kl_before = self._compute_kl_constraint(obs_flat)

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat)

        with torch.no_grad():
            policy_loss_after = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_after = self._compute_vf_loss(
                obs_flat, returns_flat)
            # kl_after = self._compute_kl_constraint(obs)
            kl_after = self._compute_kl_constraint(obs_flat)
            # policy_entropy = self._compute_policy_entropy(obs)
            policy_entropy = self._compute_policy_entropy(obs_flat)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['LossBefore'] = policy_loss_before.item()
            self.eval_statistics['LossAfter'] = policy_loss_after.item()
            self.eval_statistics['dLoss'] = (policy_loss_before - policy_loss_after).item()
            self.eval_statistics['KLBefore'] = kl_before.item()
            self.eval_statistics['KL'] = kl_after.item()
            self.eval_statistics['Entropy'] = policy_entropy.mean().item()

            self.eval_statistics['VF LossBefore'] = vf_loss_before.item()
            self.eval_statistics['VF LossAfter'] = vf_loss_after.item()
            self.eval_statistics['VF dLoss'] = (vf_loss_before - vf_loss_after).item()
            for i in range(len(sup_losses)):
                self.eval_statistics['SUP Loss {}'.format(i)] = sup_losses[i]

        self._old_policy = copy.deepcopy(self.policy)

    def _train_sup_learners(self, observations, n_labels):
        sup_losses = []
        observations = torch_ify(observations)
        for learner, labels, optimizer in zip(self.sup_learners, n_labels, self._sup_optimizers):
            labels = torch_ify(labels)
            optimizer.zero_grad()
            loss = self._compute_sup_loss(learner, observations, labels)
            loss.backward()
            optimizer.step()
            sup_losses.append(loss.item())
        return sup_losses

    def _compute_sup_loss(self, learner, obs, labels):
        valid_mask = ~torch.isnan(labels).squeeze(-1)
        lls = learner.log_prob(obs[valid_mask], labels[valid_mask])
        return -lls.mean()

    def _train(self, obs, actions, rewards, returns, advs):
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
                obs, actions, rewards, advs):
            self._train_policy(*dataset)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)

    def _train_policy(self, obs, actions, rewards, advantages):
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
        loss = self._compute_loss_with_adv(obs, actions, rewards, advantages)
        loss.backward()
        self._policy_optimizer.step()

        return loss

    def _train_value_function(self, obs, returns):
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
        loss = self._compute_vf_loss(obs, returns)
        loss.backward()
        self._vf_optimizer.step()

        return loss

    def _compute_loss_with_adv(self, obs, actions, rewards, advantages):
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

        return -objectives.mean()

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
        advantage_flat = torch.cat(filter_valids(advantages, valids))

        if self._center_adv:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat

    def _compute_kl_constraint(self, obs):
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

            return kl_constraint.mean()
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

    def _compute_vf_loss(self, obs, returns):
        baselines = self._value_function(obs).squeeze(-1)
        vf_loss = self._vf_criterion(baselines, returns)
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
        # Compute constraint
        with torch.no_grad():
            old_ll = self._old_policy.log_prob(obs, actions)
        new_ll = self.policy.log_prob(obs, actions)

        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        # Clipping the constraint
        likelihood_ratio_clip = torch.clamp(likelihood_ratio,
                                            min=1 - self._lr_clip_range,
                                            max=1 + self._lr_clip_range)

        # Calculate surrotate clip
        surrogate_clip = likelihood_ratio_clip * advantages

        return torch.min(surrogate, surrogate_clip)

    def _add_exploration_bonus(self, paths):
        paths = copy.deepcopy(paths)
        entropy_decreases = []
        with torch.no_grad():
            for path in paths:
                for i in range(len(path['observations'])-1):
                    obs1 = path['observations'][i]
                    entropy_1 = [sup_learner.get_distribution(torch_ify(obs1)).entropy() for sup_learner in self.sup_learners]
                    entropy_1 = torch.sum(torch.stack(entropy_1))
                    obs2 = path['observations'][i+1]
                    entropy_2 = [sup_learner.get_distribution(torch_ify(obs2)).entropy() for sup_learner in self.sup_learners]
                    entropy_2 = torch.sum(torch.stack(entropy_2))
                    entropy_decrease = (entropy_1 - entropy_2).item()
                    entropy_decreases.append(entropy_decrease)
                    path['rewards'][i] += self.exploration_bonus*entropy_decrease

        if self._need_to_update_eval_statistics:
            self.eval_statistics.update(create_stats_ordered_dict(
                'Entropy Decrease',
                entropy_decreases,
            ))
        return paths

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
        if self.exploration_bonus > 0.:
            paths = self._add_exploration_bonus(paths)
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
        # batch x label_num x label_dim
        n_labels = []
        for lid in range(len(paths[0]['env_infos'][0]['sup_labels'])):
            labels = []
            for path in paths:
                path_labels = []
                for info in path['env_infos']:
                    path_labels.append(info['sup_labels'][lid])
                labels.append(path_labels)
            labels = torch.stack([
                pad_to_last(path_labels,
                            total_length=self.max_path_length,
                            axis=0) for path_labels in labels
            ])
            n_labels.append(labels)

        with torch.no_grad():
            baselines = self._value_function(obs).squeeze(-1)

        return obs, actions, rewards, returns, valids, baselines, n_labels

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
            *self.sup_learners,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            old_policy=self._old_policy,
            value_function=self._value_function,
            sup_learners=self.sup_learners,
        )
