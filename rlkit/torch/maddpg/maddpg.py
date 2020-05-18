from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class MADDPGTrainer(TorchTrainer):
    """
    Deep Deterministic Policy Gradient
    """
    def __init__(
            self,
            qf_n,
            target_qf_n,
            policy_n,
            target_policy_n,
            online_action,
            double_q,
            qf2_n=None,
            target_qf2_n=None,

            discount=0.99,
            reward_scale=1.0, # not used

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            qf_criterion=None,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            min_q_value=-np.inf,
            max_q_value=np.inf,
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf_n = qf_n
        self.target_qf_n = target_qf_n
        self.policy_n = policy_n
        self.target_policy_n = target_policy_n
        self.online_action = online_action
        self.double_q = double_q
        self.qf2_n = qf2_n
        self.target_qf2_n = target_qf2_n

        self.discount = discount
        self.reward_scale = reward_scale

        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.qf_criterion = qf_criterion
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.min_q_value = min_q_value
        self.max_q_value = max_q_value

        self.qf_optimizer_n = [ 
            optimizer_class(
                self.qf_n[i].parameters(),
                lr=self.qf_learning_rate,
            ) for i in range(len(self.qf_n))]
        if self.double_q:
            self.qf2_optimizer_n = [ 
                optimizer_class(
                    self.qf2_n[i].parameters(),
                    lr=self.qf_learning_rate,
                ) for i in range(len(self.qf2_n))]

        self.policy_optimizer_n = [
            optimizer_class(
                self.policy_n[i].parameters(),
                lr=self.policy_learning_rate,
            ) for i in range(len(self.policy_n))]

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards_n = batch['rewards']
        terminals_n = batch['terminals']
        obs_n = batch['observations']
        actions_n = batch['actions']
        next_obs_n = batch['next_observations']

        batch_size = rewards_n.shape[0]
        num_agent = rewards_n.shape[1]
        whole_obs = obs_n.view(batch_size, -1)
        whole_actions = actions_n.view(batch_size, -1)
        whole_next_obs = next_obs_n.view(batch_size, -1)
        if self.online_action:
            online_actions_n = [self.policy_n[agent](obs_n[:,agent,:]).detach() for agent in range(num_agent)]
            online_actions_n = torch.stack(online_actions_n) # num_agent x batch x a_dim
            online_actions_n = online_actions_n.transpose(0,1).contiguous() # batch x num_agent x a_dim     

        next_target_actions_n = [self.target_policy_n[agent](next_obs_n[:,agent,:]).detach() for agent in range(num_agent)]
        next_target_actions_n = torch.stack(next_target_actions_n) # num_agent x batch x a_dim
        next_target_actions_n = next_target_actions_n.transpose(0,1).contiguous() # batch x num_agent x a_dim

        for agent in range(num_agent):
            """
            Policy operations.
            """
            if self.policy_pre_activation_weight > 0:
                policy_actions, pre_value = self.policy_n[agent](
                    obs_n[:,agent,:], return_info=True,
                )
                pre_value = info['preactivation']
                pre_activation_policy_loss = (
                    (pre_value**2).sum(dim=1).mean()
                )
            else:
                policy_actions = self.policy_n[agent](obs_n[:,agent,:])
                pre_activation_policy_loss = 0
            if self.online_action:
                current_actions = online_actions_n.clone()
            else:
                current_actions = actions_n.clone()
            current_actions[:,agent,:] = policy_actions 
            q_output = self.qf_n[agent](whole_obs, current_actions.view(batch_size, -1))
            if self.double_q:
                q2_output = self.qf2_n[agent](whole_obs, current_actions.view(batch_size, -1))
                q_output = torch.min(q_output,q2_output)
            raw_policy_loss = - q_output.mean()
            policy_loss = (
                    raw_policy_loss +
                    pre_activation_policy_loss * self.policy_pre_activation_weight
            )

            """
            Critic operations.
            """
            # speed up computation by not backpropping these gradients
            next_target_q_values = self.target_qf_n[agent](
                whole_next_obs,
                next_target_actions_n.view(batch_size,-1),
            )
            if self.double_q:
                next_target_q2_values = self.target_qf2_n[agent](
                    whole_next_obs,
                    next_target_actions_n.view(batch_size,-1),
                )
                next_target_q_values = torch.min(next_target_q_values, next_target_q2_values)

            q_target = self.reward_scale*rewards_n[:,agent,:] + (1. - terminals_n[:,agent,:]) * self.discount * next_target_q_values
            q_target = q_target.detach()
            q_target = torch.clamp(q_target, self.min_q_value, self.max_q_value)
            q_pred = self.qf_n[agent](whole_obs, whole_actions)
            bellman_errors = (q_pred - q_target) ** 2
            raw_qf_loss = self.qf_criterion(q_pred, q_target)
            if self.qf_weight_decay > 0:
                reg_loss = self.qf_weight_decay * sum(
                    torch.sum(param ** 2)
                    for param in self.qf_n[agent].regularizable_parameters()
                )
                qf_loss = raw_qf_loss + reg_loss
            else:
                qf_loss = raw_qf_loss

            if self.double_q:
                q2_pred = self.qf2_n[agent](whole_obs, whole_actions)
                bellman_errors2 = (q2_pred - q_target) ** 2
                raw_qf2_loss = self.qf_criterion(q2_pred, q_target)
                if self.qf_weight_decay > 0:
                    reg_loss2 = self.qf_weight_decay * sum(
                        torch.sum(param ** 2)
                        for param in self.qf2_n[agent].regularizable_parameters()
                    )
                    qf2_loss = raw_qf2_loss + reg_loss2
                else:
                    qf2_loss = raw_qf2_loss

            """
            Update Networks
            """

            self.policy_optimizer_n[agent].zero_grad()
            policy_loss.backward()
            self.policy_optimizer_n[agent].step()

            self.qf_optimizer_n[agent].zero_grad()
            qf_loss.backward()
            self.qf_optimizer_n[agent].step()

            if self.double_q:
                self.qf2_optimizer_n[agent].zero_grad()
                qf2_loss.backward()
                self.qf2_optimizer_n[agent].step()
            """
            Save some statistics for eval using just one batch.
            """
            if self._need_to_update_eval_statistics:
                self.eval_statistics['QF Loss {}'.format(agent)] = np.mean(ptu.get_numpy(qf_loss))
                if self.double_q:
                    self.eval_statistics['QF2 Loss {}'.format(agent)] = np.mean(ptu.get_numpy(qf2_loss))
                self.eval_statistics['Policy Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    policy_loss
                ))
                self.eval_statistics['Raw Policy Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    raw_policy_loss
                ))
                self.eval_statistics['Preactivation Policy Loss {}'.format(agent)] = (
                        self.eval_statistics['Policy Loss {}'.format(agent)] -
                        self.eval_statistics['Raw Policy Loss {}'.format(agent)]
                )
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Predictions {}'.format(agent),
                    ptu.get_numpy(q_pred),
                ))
                if self.double_q:
                    self.eval_statistics.update(create_stats_ordered_dict(
                        'Q2 Predictions {}'.format(agent),
                        ptu.get_numpy(q2_pred),
                    ))   
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Targets {}'.format(agent),
                    ptu.get_numpy(q_target),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Bellman Errors {}'.format(agent),
                    ptu.get_numpy(bellman_errors),
                ))
                if self.double_q:
                    self.eval_statistics.update(create_stats_ordered_dict(
                        'Bellman Errors2 {}'.format(agent),
                        ptu.get_numpy(bellman_errors2),
                    ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy Action {}'.format(agent),
                    ptu.get_numpy(policy_actions),
                ))
                
        self._need_to_update_eval_statistics = False
        self._update_target_networks()
        self._n_train_steps_total += 1

    def _update_target_networks(self):
        for policy, target_policy, qf, target_qf in \
            zip(self.policy_n, self.target_policy_n, self.qf_n, self.target_qf_n):
            if self.use_soft_update:
                ptu.soft_update_from_to(policy, target_policy, self.tau)
                ptu.soft_update_from_to(qf, target_qf, self.tau)
            else:
                if self._n_train_steps_total % self.target_hard_update_period == 0:
                    ptu.copy_model_params_from_to(qf, target_qf)
                    ptu.copy_model_params_from_to(policy, target_policy)
        if self.double_q:
            for qf2, target_qf2 in zip(self.qf2_n, self.target_qf2_n):
                if self.use_soft_update:
                    ptu.soft_update_from_to(qf2, target_qf2, self.tau)
                else:
                    if self._n_train_steps_total % self.target_hard_update_period == 0:
                        ptu.copy_model_params_from_to(qf2, target_qf2)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        res = [
            *self.policy_n,
            *self.qf_n,
            *self.target_policy_n,
            *self.target_qf_n,
        ]
        if self.double_q:
            res += [*self.qf2_n, *self.target_qf2_n]
        return res

    def get_snapshot(self):
        res = dict(
            qf_n=self.qf_n,
            target_qf_n=self.target_qf_n,
            trained_policy_n=self.policy_n,
            target_policy_n=self.target_policy_n,
        )
        if self.double_q:
            res['qf2_n'] = self.qf2_n
            res['target_qf2_n'] = self.target_qf2_n
        return res
