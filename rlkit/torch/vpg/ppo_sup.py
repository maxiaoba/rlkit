import copy
import torch
from collections import OrderedDict

from rlkit.util import tensor_util as tu
from rlkit.torch.vpg.ppo import PPOTrainer
from rlkit.torch.vpg.util import compute_advantages, filter_valids, pad_to_last
from rlkit.torch.core import torch_ify
from rlkit.core.eval_util import create_stats_ordered_dict

class PPOSupTrainer(PPOTrainer):
    """PPO + supervised learning.
    """

    def __init__(self,
                 sup_learner,
                 replay_buffer,
                 exploration_bonus,
                 sup_lr=1e-3,
                 sup_batch_size=64,
                 sup_train_num=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.sup_learner = sup_learner
        self.replay_buffer = replay_buffer
        self.sup_batch_size = sup_batch_size
        self.sup_train_num = sup_train_num
        self.exploration_bonus = exploration_bonus
        self._sup_optimizer = torch.optim.Adam(
                                self.sup_learner.parameters(),
                                lr=sup_lr)

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
            policy_loss_before = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_before = self._compute_vf_loss(
                obs_flat, returns_flat)
            # kl_before = self._compute_kl_constraint(obs)
            kl_before = self._compute_kl_constraint(obs_flat)

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat)

        self.replay_buffer.add_batch(obs_flat, labels_flat)
        for _ in range(self.sup_train_num):
            batch = self.replay_buffer.random_batch(self.sup_batch_size)
            sup_loss = self._train_sup_learner(batch['observations'],batch['labels'])

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
            self.eval_statistics['SUP Loss'] = sup_loss.item()

        self._old_policy = copy.deepcopy(self.policy)

    def _train_sup_learner(self, observations, labels):
        observations = torch_ify(observations)
        labels = torch_ify(labels)
        self._sup_optimizer.zero_grad()
        sup_loss = self._compute_sup_loss(observations, labels)
        sup_loss.backward()
        self._sup_optimizer.step()
        return sup_loss

    def _compute_sup_loss(self, obs, labels):
        obs = torch_ify(obs)
        labels = torch_ify(labels).clone()
        valid_mask = ~torch.isnan(labels) # replay buffer!
        labels[~valid_mask] = 0     
        lls = self.sup_learner.log_prob(obs, labels)
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
                    entropy_1 = self.sup_learner.get_distribution(torch_ify(obs1)[None,:]).entropy()
                    entropy_1 = torch.mean(entropy_1[valid_mask1])

                    obs2 = path['observations'][i+1]
                    labels2 = torch.tensor(path['env_infos']['sup_labels'][i+1])
                    valid_mask2 = ~torch.isnan(labels2)[None,:]
                    entropy_2 = self.sup_learner.get_distribution(torch_ify(obs2)[None,:]).entropy()
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

    @property
    def networks(self):
        return [
            self._value_function,
            self._old_policy,
            self.policy,
            self.sup_learner,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            old_policy=self._old_policy,
            value_function=self._value_function,
            sup_learner=self.sup_learner,
        )
