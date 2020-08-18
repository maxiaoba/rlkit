import copy
import torch
from collections import OrderedDict

from rlkit.util import tensor_util as tu
from rlkit.torch.vpg.ppo import PPOTrainer
from rlkit.torch.vpg.util import compute_advantages, filter_valids, pad_to_last
from rlkit.torch.core import torch_ify
from rlkit.core.eval_util import create_stats_ordered_dict

class PPOSupOnlineTrainer(PPOTrainer):
    """PPO + supervised learning with online data.
    """

    def __init__(self,
                 sup_weight,
                 exploration_bonus,
                 **kwargs):
        super().__init__(**kwargs)
        self.sup_weight = sup_weight
        self.exploration_bonus = exploration_bonus

    def train_once(self, paths):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """

        obs, actions, rewards, returns, valids, baselines, labels = \
            self.process_samples(paths)

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = torch.cat(filter_valids(obs, valids))
        actions_flat = torch.cat(filter_valids(actions, valids))
        rewards_flat = torch.cat(filter_valids(rewards, valids))
        returns_flat = torch.cat(filter_valids(returns, valids))
        advs_flat = self._compute_advantage(rewards, valids, baselines)
        labels_flat = torch.cat(filter_valids(labels, valids))

        with torch.no_grad():
            sup_loss_before = self._compute_sup_loss(obs_flat, labels_flat)
            policy_loss_before = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat, labels_flat)
            vf_loss_before = self._compute_vf_loss(
                obs_flat, returns_flat)
            # kl_before = self._compute_kl_constraint(obs)
            kl_before = self._compute_kl_constraint(obs_flat)

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat, labels_flat)

        with torch.no_grad():
            sup_loss_after = self._compute_sup_loss(obs_flat, labels_flat)
            policy_loss_after = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat, labels_flat)
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

            self.eval_statistics['SUP LossBefore'] = sup_loss_before.item()
            self.eval_statistics['SUP LossAfter'] = sup_loss_after.item()
            self.eval_statistics['SUP dLoss'] = (sup_loss_before - sup_loss_after).item()

        self._old_policy = copy.deepcopy(self.policy)

    def _train(self, obs, actions, rewards, returns, advs, labels):
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
            labels (torch.Tensor): Sup labels
                :math:`(N, label_dim)`.
        """
        for dataset in self._policy_optimizer.get_minibatch(
                obs, actions, rewards, advs, labels):
            self._train_policy(*dataset)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)

    def _train_policy(self, obs, actions, rewards, advantages, labels):
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
        loss = self._compute_loss_with_adv(obs, actions, rewards, advantages, labels)
        loss.backward()
        self._policy_optimizer.step()

        return loss


    def _compute_loss_with_adv(self, obs, actions, rewards, advantages, labels):
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
        objectives = self._compute_objective(advantages, obs, actions, rewards, labels)

        if self._entropy_regularzied:
            policy_entropies = self._compute_policy_entropy(obs)
            objectives += self._policy_ent_coeff * policy_entropies

        return -objectives.mean()

    def _compute_objective(self, advantages, obs, actions, rewards, labels):
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
        sup_loss = self._compute_sup_loss(obs, labels)
        surrogate = likelihood_ratio * (advantages - self.sup_weight*sup_loss)

        # Clipping the constraint
        likelihood_ratio_clip = torch.clamp(likelihood_ratio,
                                            min=1 - self._lr_clip_range,
                                            max=1 + self._lr_clip_range)

        # Calculate surrotate clip
        surrogate_clip = likelihood_ratio_clip * advantages

        return torch.min(surrogate, surrogate_clip)

    def _compute_sup_loss(self, obs, labels):
        obs = torch_ify(obs)
        labels = torch_ify(labels)
        valid_mask = ~torch.isnan(labels) # replay buffer!
        labels[~valid_mask] = 0     
        lls = self.policy.sup_log_prob(obs, labels)
        return -lls[valid_mask].mean()

    def _add_exploration_bonus(self, paths):
        paths = copy.deepcopy(paths)
        entropy_decreases = []
        with torch.no_grad():
            for path in paths:
                for i in range(len(path['observations'])-1):
                    obs1 = path['observations'][i]
                    labels1 = torch.tensor(path['env_infos']['sup_labels'][i])
                    valid_mask1 = ~torch.isnan(labels1)[None,:]
                    entropy_1 = self.policy.get_sup_distribution(torch_ify(obs1)[None,:]).entropy()
                    entropy_1 = torch.mean(entropy_1[valid_mask1])

                    obs2 = path['observations'][i+1]
                    labels2 = torch.tensor(path['env_infos']['sup_labels'][i+1])
                    valid_mask2 = ~torch.isnan(labels2)[None,:]
                    entropy_2 = self.policy.get_sup_distribution(torch_ify(obs2)[None,:]).entropy()
                    entropy_2 = torch.mean(entropy_2[valid_mask2])

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
        labels = torch.stack([
            pad_to_last(path['env_infos']['sup_labels'],
                        total_length=self.max_path_length,
                        axis=0) for path in paths
        ])

        with torch.no_grad():
            baselines = self._value_function(obs).squeeze(-1)

        return obs, actions, rewards, returns, valids, baselines, labels
