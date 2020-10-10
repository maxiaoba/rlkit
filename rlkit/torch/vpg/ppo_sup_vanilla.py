import copy
import torch
from collections import OrderedDict

from rlkit.util import tensor_util as tu
from rlkit.torch.vpg.ppo import PPOTrainer
from rlkit.torch.vpg.util import compute_advantages, filter_valids, pad_to_last
from rlkit.torch.core import torch_ify
from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.pythonplusplus as ppp

class PPOSupVanillaTrainer(PPOTrainer):
    """PPO + supervised learning.
    """

    def __init__(self,
                 replay_buffer,
                 exploration_bonus,
                 sup_lr=1e-3,
                 sup_batch_size=64,
                 sup_train_num=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.replay_buffer = replay_buffer
        self.sup_batch_size = sup_batch_size
        self.sup_train_num = sup_train_num
        self.exploration_bonus = exploration_bonus
        self._sup_optimizer = torch.optim.Adam(
                                self.policy.parameters(),
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
        advs = self._compute_advantage(rewards, valids, baselines)

        if self._recurrent:
            pre_actions = actions[:,:-1,:]
            policy_input = (obs,pre_actions)
            obs_input, actions_input, rewards_input, returns_input, advs_input = \
                obs, actions, rewards, returns, advs
            labels_input = labels
            valid_mask = torch.zeros(obs.shape[0],obs.shape[1]).bool()
            for i, valid in enumerate(valids):
                valid_mask[i,:valid] = True
        else:
            obs_input = torch.cat(filter_valids(obs, valids))
            actions_input = torch.cat(filter_valids(actions, valids))
            rewards_input = torch.cat(filter_valids(rewards, valids))
            returns_input = torch.cat(filter_valids(returns, valids))
            advs_input = torch.cat(filter_valids(advs, valids))
            labels_input = torch.cat(filter_valids(labels, valids))
            policy_input = obs_input
            valid_mask = torch.ones(obs_input.shape[0]).bool()
            # (num of valid samples) x ...
        self.replay_buffer.add_batch(obs_input, actions_input, labels_input, valid_mask)

        with torch.no_grad():
            policy_loss_before = self._compute_loss_with_adv(
                policy_input, actions_input, rewards_input, advs_input, valid_mask)
            vf_loss_before = self._compute_vf_loss(
                obs_input, returns_input, valid_mask)
            # kl_before = self._compute_kl_constraint(obs)
            kl_before = self._compute_kl_constraint(policy_input, valid_mask)
            sup_loss_before, sup_accuracy_before = self._compute_sup_loss(obs_input, actions_input, labels_input, valid_mask)
            
        self._train(policy_input, obs_input, actions_input, rewards_input, returns_input,
                    advs_input, valid_mask)

        for _ in range(self.sup_train_num):
            sup_batch = self.replay_buffer.random_batch(self.sup_batch_size)
            sup_loss = self._train_sup_learner(sup_batch['observations'],sup_batch['actions'],
                                                sup_batch['labels'],sup_batch['valids'])

        with torch.no_grad():
            policy_loss_after = self._compute_loss_with_adv(
                policy_input, actions_input, rewards_input, advs_input, valid_mask)
            vf_loss_after = self._compute_vf_loss(
                obs_input, returns_input, valid_mask)
            # kl_before = self._compute_kl_constraint(obs)
            kl_after = self._compute_kl_constraint(policy_input, valid_mask)
            sup_loss_after, sup_accuracy_after = self._compute_sup_loss(obs_input, actions_input, labels_input, valid_mask)
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

            self.eval_statistics['SUP LossBefore'] = sup_loss_before.item()
            self.eval_statistics['SUP LossAfter'] = sup_loss_after.item()
            self.eval_statistics['SUP dLoss'] = (sup_loss_before - sup_loss_after).item()
            self.eval_statistics['SUP AccuracyBefore'] = sup_accuracy_before.item()
            self.eval_statistics['SUP AccuracyAfter'] = sup_accuracy_after.item()
            self.eval_statistics['SUP dAccuracy'] = (sup_accuracy_before - sup_accuracy_after).item()

        self._old_policy = copy.deepcopy(self.policy)

    def _train_sup_learner(self, observations, actions, labels, valids):
        self._sup_optimizer.zero_grad()
        sup_loss, sup_accuracy = self._compute_sup_loss(observations, actions, labels, valids)
        sup_loss.backward()
        self._sup_optimizer.step()
        return sup_loss

    def _compute_sup_loss(self, obs, actions, labels, valid_mask):
        obs = torch_ify(obs)
        actions = torch_ify(actions)
        valid_mask = torch_ify(valid_mask).bool()
        labels = torch_ify(labels).clone()
        valids = ~torch.isnan(labels)
        labels[~valids] = 0
        if self._recurrent:
            pre_actions = actions[:,:-1,:]  
            policy_input = (obs, pre_actions)
        else:
            policy_input = obs     
        # lls = self.policy.sup_log_prob(policy_input, labels)

        valid_num = (valid_mask.unsqueeze(-1)*valids).float().sum()
        dists =  self.policy.get_sup_distribution(policy_input)
        lls =  dists.log_prob(labels)
        lls[~valids] = 0
        lls[~valid_mask] = 0
        loss = -lls.sum()/valid_num

        accuracy = (torch.argmax(dists.probs,-1)==labels).float()
        accuracy[~valids] = 0.
        accuracy[~valid_mask] = 0.
        accuracy = accuracy.sum()/valid_num

        # return -lls[valid_mask].mean()
        return loss, accuracy

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
                    # if self.attention_eb: # todo
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
        env_infos = [ppp.list_of_dicts__to__dict_of_lists(p['env_infos']) for p in paths]
        labels = torch.stack([
            pad_to_last(env_info['sup_labels'],
                        total_length=self.max_path_length,
                        axis=0) for env_info in env_infos
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
            # self.sup_learner,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            old_policy=self._old_policy,
            value_function=self._value_function,
            # sup_learner=self.sup_learner,
        )
