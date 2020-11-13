from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class MASACTrainer(TorchTrainer):
    """
    Soft Actor Critic
    """
    def __init__(
            self,
            env,
            qf1_n,
            target_qf1_n,
            qf2_n,
            target_qf2_n,
            policy_n,
            online_action,

            discount=0.99,
            reward_scale=1.0,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0.,
            init_alpha=1.,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            qf_criterion=None,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            optimizer_class=optim.Adam,

            min_q_value=-np.inf,
            max_q_value=np.inf,
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.env = env
        self.qf1_n = qf1_n
        self.target_qf1_n = target_qf1_n
        self.qf2_n = qf2_n
        self.target_qf2_n = target_qf2_n
        self.policy_n = policy_n
        self.online_action = online_action

        self.discount = discount
        self.reward_scale = reward_scale

        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.qf_criterion = qf_criterion
        self.min_q_value = min_q_value
        self.max_q_value = max_q_value

        self.init_alpha = init_alpha
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            # self.log_alpha_n = [ptu.zeros(1, requires_grad=True) for i in range(len(self.policy_n))]
            self.log_alpha_n = [ptu.tensor([np.log(self.init_alpha)], requires_grad=True, dtype=torch.float32) for i in range(len(self.policy_n))]
            self.alpha_optimizer_n = [
                optimizer_class(
                    [self.log_alpha_n[i]],
                    lr=self.policy_learning_rate,
                ) for i in range(len(self.log_alpha_n))]

        self.qf1_optimizer_n = [ 
            optimizer_class(
                self.qf1_n[i].parameters(),
                lr=self.qf_learning_rate,
            ) for i in range(len(self.qf1_n))]
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

        with torch.no_grad():
            if self.online_action:
                online_actions_n = [self.policy_n[agent](obs_n[:,agent,:]) for agent in range(num_agent)]
                online_actions_n = torch.stack(online_actions_n) # num_agent x batch x a_dim
                online_actions_n = online_actions_n.transpose(0,1).contiguous() # batch x num_agent x a_dim     

            next_actions_n = [self.policy_n[agent](next_obs_n[:,agent,:]) for agent in range(num_agent)]
            next_actions_n = torch.stack(next_actions_n) # num_agent x batch x a_dim
            next_actions_n = next_actions_n.transpose(0,1).contiguous() # batch x num_agent x a_dim

        for agent in range(num_agent):
            """
            Policy operations.
            """
            policy_actions, info = self.policy_n[agent](
                obs_n[:,agent,:], return_info=True,
            )
            log_pi = info['log_prob']
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha_n[agent].exp() * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer_n[agent].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer_n[agent].step()
                alpha = self.log_alpha_n[agent].exp()
            else:
                alpha_loss = torch.tensor(0.).to(ptu.device)
                alpha = torch.tensor(self.init_alpha).to(ptu.device)

            if self.online_action:
                current_actions = online_actions_n.clone()
            else:
                current_actions = actions_n.clone()
            current_actions[:,agent,:] = policy_actions 
            q1_output = self.qf1_n[agent](whole_obs, current_actions.view(batch_size, -1))
            q2_output = self.qf2_n[agent](whole_obs, current_actions.view(batch_size, -1))
            min_q_output = torch.min(q1_output,q2_output)
            policy_loss = (alpha*log_pi - min_q_output).mean()

            self.policy_optimizer_n[agent].zero_grad()
            policy_loss.backward()
            self.policy_optimizer_n[agent].step()

            """
            Critic operations.
            """
            with torch.no_grad():
                new_actions, new_info = self.policy_n[agent](
                    next_obs_n[:,agent,:], return_info=True,
                )
                new_log_pi = new_info['log_prob']
                new_actions = new_actions
                new_log_pi = new_log_pi
                next_current_actions = next_actions_n.clone()
                next_current_actions[:,agent,:] = new_actions
                next_target_q1_values = self.target_qf1_n[agent](
                    whole_next_obs,
                    next_current_actions.view(batch_size,-1),
                )
                next_target_q2_values = self.target_qf2_n[agent](
                    whole_next_obs,
                    next_current_actions.view(batch_size,-1),
                )
                next_target_min_q_values = torch.min(next_target_q1_values,next_target_q2_values)
                next_target_q_values =  next_target_min_q_values - alpha * new_log_pi
                q_target = self.reward_scale*rewards_n[:,agent,:] + (1. - terminals_n[:,agent,:]) * self.discount * next_target_q_values
                q_target = torch.clamp(q_target, self.min_q_value, self.max_q_value)

            q1_pred = self.qf1_n[agent](whole_obs, whole_actions)
            q2_pred = self.qf2_n[agent](whole_obs, whole_actions)

            qf1_loss = self.qf_criterion(q1_pred, q_target)
            qf2_loss = self.qf_criterion(q2_pred, q_target)

            self.qf1_optimizer_n[agent].zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer_n[agent].step()

            self.qf2_optimizer_n[agent].zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer_n[agent].step()

            """
            Save some statistics for eval using just one batch.
            """
            if self._need_to_update_eval_statistics:
                self.eval_statistics['QF1 Loss {}'.format(agent)] = np.mean(ptu.get_numpy(qf1_loss))
                self.eval_statistics['QF2 Loss {}'.format(agent)] = np.mean(ptu.get_numpy(qf2_loss))
                self.eval_statistics['Policy Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    policy_loss
                ))
                self.eval_statistics['Alpha Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    alpha_loss
                ))
                self.eval_statistics['Alpha {}'.format(agent)] = np.mean(ptu.get_numpy(
                    alpha
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q1 Predictions {}'.format(agent),
                    ptu.get_numpy(q1_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions {}'.format(agent),
                    ptu.get_numpy(q2_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Targets {}'.format(agent),
                    ptu.get_numpy(q_target),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy Action {}'.format(agent),
                    ptu.get_numpy(policy_actions),
                ))
                
        self._need_to_update_eval_statistics = False
        self._update_target_networks()
        self._n_train_steps_total += 1

    def _update_target_networks(self):
        for qf1, target_qf1, qf2, target_qf2 in \
            zip(self.qf1_n, self.target_qf1_n, self.qf2_n, self.target_qf2_n):
            if self.use_soft_update:
                ptu.soft_update_from_to(qf1, target_qf1, self.tau)
                ptu.soft_update_from_to(qf2, target_qf2, self.tau)
            else:
                if self._n_train_steps_total % self.target_hard_update_period == 0:
                    ptu.copy_model_params_from_to(qf1, target_qf1)
                    ptu.copy_model_params_from_to(qf2, target_qf2)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            *self.policy_n,
            *self.qf1_n,
            *self.qf2_n,
            *self.target_qf1_n,
            *self.target_qf2_n,
        ]

    def get_snapshot(self):
        return dict(
            qf1_n=self.qf1_n,
            target_qf1_n=self.target_qf1_n,
            qf2_n=self.qf2_n,
            target_qf2_n=self.target_qf2_n,
            trained_policy_n=self.policy_n,
        )
