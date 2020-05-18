from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class MASACDiscreteTrainer(TorchTrainer):
    """
    Soft Actor Critic on Discrete Action Space
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
            clip_gradient=0.,

            discount=0.99,
            reward_scale=1.0,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0,
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
            # qf_criterion = nn.SmoothL1Loss() # Huber Loss
        self.env = env
        self.qf1_n = qf1_n
        self.target_qf1_n = target_qf1_n
        self.qf2_n = qf2_n
        self.target_qf2_n = target_qf2_n
        self.policy_n = policy_n
        self.online_action = online_action
        self.clip_gradient = clip_gradient

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

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = 0.5*np.log(self.env.action_space.n)  # heuristic value
            self.log_alpha_n = [ptu.zeros(1, requires_grad=True) for i in range(len(self.policy_n))]
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
        if self.online_action:
            online_actions_n = [self.policy_n[agent].one_hot(obs_n[:,agent,:]).detach() for agent in range(num_agent)]
            online_actions_n = torch.stack(online_actions_n) # num_agent x batch x a_dim
            online_actions_n = online_actions_n.transpose(0,1).contiguous() # batch x num_agent x a_dim     

        next_actions_n = [self.policy_n[agent].one_hot(next_obs_n[:,agent,:]).detach() for agent in range(num_agent)]
        next_actions_n = torch.stack(next_actions_n) # num_agent x batch x a_dim
        next_actions_n = next_actions_n.transpose(0,1).contiguous() # batch x num_agent x a_dim

        for agent in range(num_agent):
            """
            Policy operations.
            """
            pis = self.policy_n[agent](obs_n[:,agent,:]) # batch x |A|

            if self.use_automatic_entropy_tuning:
                alpha_loss = -(pis.detach() * self.log_alpha_n[agent].exp() * (torch.log(pis) + self.target_entropy).detach()).sum(-1).mean()
                self.alpha_optimizer_n[agent].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer_n[agent].step()
                alpha = self.log_alpha_n[agent].exp()
            else:
                alpha_loss = torch.tensor(0.)
                alpha = torch.tensor(1.)

            if self.online_action:
                current_actions = online_actions_n.clone()
            else:
                current_actions = actions_n.clone()
            other_action_index = np.array([i for i in range(num_agent) if i!=agent])
            other_actions = current_actions[:,other_action_index,:]
            q1_output = self.qf1_n[agent](whole_obs, other_actions.view(batch_size, -1))
            q2_output = self.qf2_n[agent](whole_obs, other_actions.view(batch_size, -1))
            min_q_output = torch.min(q1_output,q2_output) # batch x |A|
            policy_loss = (pis*(alpha*torch.log(pis) - min_q_output)).sum(-1).mean()
            # policy_loss = (pis*(torch.log(pis)-torch.log(torch.softmax(min_q_output/alpha, dim=-1)))).sum(-1).mean()

            """
            Critic operations.
            """
            # speed up computation by not backpropping these gradients
            new_pis = self.policy_n[agent](next_obs_n[:,agent,:]).detach()
            next_current_actions = next_actions_n.clone()
            next_other_actions = next_current_actions[:,other_action_index,:].detach()
            other_actions = actions_n[:,other_action_index,:].detach()
            next_target_q1_values = self.target_qf1_n[agent](
                whole_next_obs,
                next_other_actions.view(batch_size,-1),
            )
            next_target_q2_values = self.target_qf2_n[agent](
                whole_next_obs,
                next_other_actions.view(batch_size,-1),
            )
            next_target_min_q_values = torch.min(next_target_q1_values,next_target_q2_values) # batch x |A|
            next_target_q_values =  (new_pis*(next_target_min_q_values - alpha * torch.log(new_pis))).sum(-1,keepdim=True) # batch
            q_target = self.reward_scale*rewards_n[:,agent,:] + (1. - terminals_n[:,agent,:]) * self.discount * next_target_q_values
            q_target = q_target.detach()
            q_target = torch.clamp(q_target, self.min_q_value, self.max_q_value)

            q1_pred = torch.sum(self.qf1_n[agent](whole_obs, other_actions.view(batch_size,-1))*actions_n[:,agent,:].detach(),dim=-1,keepdim=True)
            q2_pred = torch.sum(self.qf2_n[agent](whole_obs, other_actions.view(batch_size,-1))*actions_n[:,agent,:].detach(),dim=-1,keepdim=True)

            qf1_loss = self.qf_criterion(q1_pred, q_target)
            qf2_loss = self.qf_criterion(q2_pred, q_target)
            """
            Update Networks
            """

            self.policy_optimizer_n[agent].zero_grad()
            policy_loss.backward()
            if self.clip_gradient > 0.:
                nn.utils.clip_grad_norm_(self.policy_n[agent].parameters(), self.clip_gradient)
            self.policy_optimizer_n[agent].step()

            self.qf1_optimizer_n[agent].zero_grad()
            qf1_loss.backward()
            if self.clip_gradient > 0.:
                nn.utils.clip_grad_norm_(self.qf1_n[agent].parameters(), self.clip_gradient)
            self.qf1_optimizer_n[agent].step()

            self.qf2_optimizer_n[agent].zero_grad()
            qf2_loss.backward()
            if self.clip_gradient > 0.:
                nn.utils.clip_grad_norm_(self.qf2_n[agent].parameters(), self.clip_gradient)
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
                    'Pis {}'.format(agent),
                    ptu.get_numpy(pis),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Alpha {}'.format(agent),
                    ptu.get_numpy(alpha),
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
