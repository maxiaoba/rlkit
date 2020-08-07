from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.core import np_ify

class DQNTrainer(TorchTrainer):
    def __init__(
            self,
            qf,
            target_qf,
            double_dqn=True,
            prioritized_replay=True,
            replay_buffer=None,
            clip_gradient=0.,
            learning_rate=1e-3,
            soft_target_tau=1e-3,
            target_update_period=1,
            qf_criterion=None,

            discount=0.99,
            reward_scale=1.0,
    ):
        super().__init__()
        self.qf = qf
        self.target_qf = target_qf
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.replay_buffer = replay_buffer
        self.clip_gradient = clip_gradient
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards'] * self.reward_scale
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        if self.prioritized_replay:
            indices = batch['indices']
            importance_weights = batch['importance_weights']
        """
        Compute loss
        """
        if self.double_dqn:
            best_action_idxs = self.qf(next_obs).max(
                1, keepdim=True
            )[1]
            target_q_values = self.target_qf(next_obs).gather(
                1, best_action_idxs
            ).detach()
        else:
            target_q_values = self.target_qf(next_obs).detach().max(
                1, keepdim=True
            )[0]
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        if self.prioritized_replay:
            td_errors = y_pred - y_target
            importance_weights = importance_weights.reshape(y_pred.shape[0],1)
            # print(torch.max(importance_weights),torch.min(importance_weights),torch.mean(importance_weights))
            qf_loss = self.qf_criterion(importance_weights*y_pred,
                                        importance_weights*y_target)
            td_errors = td_errors.reshape(y_pred.shape[0])
            self.replay_buffer.update_priority(np_ify(indices).astype(int), np_ify(td_errors))
        else:
            qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Soft target network updates
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        if self.clip_gradient > 0.:
            nn.utils.clip_grad_norm_(self.qf.parameters(), self.clip_gradient)
        qf_grad_norm = torch.tensor(0.).to(ptu.device) 
        for p in self.qf.parameters():
            param_norm = p.grad.data.norm(2)
            qf_grad_norm += param_norm.item() ** 2
        qf_grad_norm = qf_grad_norm ** (1. / 2)
        self.qf_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf, self.target_qf, self.soft_target_tau
            )

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))
            self.eval_statistics['QF Gradient'] = np.mean(ptu.get_numpy(
                qf_grad_norm
            ))

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
        ]

    def get_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
        )
