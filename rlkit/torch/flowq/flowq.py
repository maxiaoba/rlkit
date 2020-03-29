from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class FlowQTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            vf1,
            vf2,
            target_vf1,
            target_vf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            vf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            clip_gradient=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.vf1 = vf1
        self.vf2 = vf2
        self.target_vf1 = target_vf1
        self.target_vf2 = target_vf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.clip_gradient = clip_gradient

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.vf_criterion = nn.MSELoss()
        self.pi_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.vf1_optimizer = optimizer_class(
            self.vf1.parameters(),
            lr=vf_lr,
        )
        self.vf2_optimizer = optimizer_class(
            self.vf2.parameters(),
            lr=vf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        raw_actions = batch['raw_actions']
        next_obs = batch['next_observations']
        """
        Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = torch.tensor(1.)

        """
        Policy and VF Loss
        """
        v1_pred = self.vf1(obs)
        v2_pred = self.vf2(obs)
        pi_pred = self.policy.log_prob(obs, actions, raw_actions)

        target_v_values = torch.min(
            self.target_vf1(next_obs),
            self.target_vf2(next_obs),
        )
        # target_v_values = self.target_vf1(next_obs)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_v_values

        vf1_loss = self.vf_criterion(v1_pred, q_target.detach()-(alpha*pi_pred).detach())
        vf2_loss = self.vf_criterion(v2_pred, q_target.detach()-(alpha*pi_pred).detach())
        policy_loss = self.pi_criterion(alpha.detach()*pi_pred, q_target.detach()-torch.min(v1_pred,v2_pred).detach())
        # policy_loss = self.pi_criterion(alpha.detach()*pi_pred, q_target.detach()-v1_pred.detach())
        """
        Update networks
        """
        self.vf1_optimizer.zero_grad()
        vf1_loss.backward()
        if self.clip_gradient:
            nn.utils.clip_grad_norm_(self.vf1.parameters(), self.clip_gradient)
        self.vf1_optimizer.step()

        self.vf2_optimizer.zero_grad()
        vf2_loss.backward()
        if self.clip_gradient:
            nn.utils.clip_grad_norm_(self.vf2.parameters(), self.clip_gradient)
        self.vf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.clip_gradient:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_gradient)
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.vf1, self.target_vf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.vf2, self.target_vf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """

            self.eval_statistics['VF1 Loss'] = np.mean(ptu.get_numpy(vf1_loss))
            self.eval_statistics['VF2 Loss'] = np.mean(ptu.get_numpy(vf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V1 Predictions',
                ptu.get_numpy(v1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V2 Predictions',
                ptu.get_numpy(v2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.vf1,
            self.vf2,
            self.target_vf1,
            self.target_vf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            vf1=self.vf1,
            vf2=self.vf2,
            target_vf1=self.vf1,
            target_vf2=self.vf2,
        )

