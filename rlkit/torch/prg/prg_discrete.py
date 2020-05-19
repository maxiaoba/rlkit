from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.nn import functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class PRGDiscreteTrainer(TorchTrainer):
    """
    Probalistic Recursive Graph
    """
    def __init__(
            self,
            env,
            qf_n,
            target_qf_n,
            policy_n,
            double_q,
            qf2_n=None,
            target_qf2_n=None,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            gumbel_hard=False,
            clip_gradient=0.,

            logit_level=1,

            discount=0.99,
            reward_scale=1.0,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            qf_criterion=None,
            pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            min_q_value=-np.inf,
            max_q_value=np.inf,
    ):
        super().__init__()
        self.env = env
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf_n = qf_n
        self.target_qf_n = target_qf_n
        self.policy_n = policy_n
        self.double_q = double_q
        self.qf2_n = qf2_n
        self.target_qf2_n = target_qf2_n

        self.logit_level = logit_level
        self.gumbel_hard = gumbel_hard
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
        self.pre_activation_weight = pre_activation_weight
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

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = 0.5*np.log(self.env.action_space.n)  # heuristic value from Tuomas
            self.log_alpha_n = [ptu.zeros(1, requires_grad=True) for i in range(len(self.policy_n))]
            self.alpha_optimizer_n = [
                optimizer_class(
                    [self.log_alpha_n[i]],
                    lr=self.policy_learning_rate,
                ) for i in range(len(self.log_alpha_n))]

        num_agent = len(self.policy_n)
        self.other_action_indices = [np.array([i for i in range(num_agent) if i!=agent]) for agent in range(num_agent)]
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards_n = batch['rewards'].detach()
        terminals_n = batch['terminals'].detach()
        obs_n = batch['observations'].detach()
        actions_n = batch['actions'].detach()
        next_obs_n = batch['next_observations'].detach()

        batch_size = rewards_n.shape[0]
        num_agent = rewards_n.shape[1]
        whole_obs = obs_n.view(batch_size, -1)
        whole_actions = actions_n.view(batch_size, -1)
        whole_next_obs = next_obs_n.view(batch_size, -1) 

        online_actions_n = [self.policy_n[agent].one_hot(obs_n[:,agent,:]).detach() for agent in range(num_agent)]
        online_actions_n = torch.stack(online_actions_n) # num_agent x batch x a_dim
        online_actions_n = online_actions_n.transpose(0,1).contiguous() # batch x num_agent x a_dim

        next_actions_n = [self.policy_n[agent].one_hot(next_obs_n[:,agent,:]).detach() for agent in range(num_agent)]
        next_actions_n = torch.stack(next_actions_n) # num_agent x batch x a_dim
        next_actions_n = next_actions_n.transpose(0,1).contiguous() # batch x num_agent x a_dim

        if self.use_automatic_entropy_tuning:
            alpha_n = [self.log_alpha_n[i].exp() for i in range(num_agent)]
        else:
            alpha_n = [1. for i in range(num_agent)]
        for agent in range(num_agent):
            """
            Policy operations.
            """
            pis, info = self.policy_n[agent](obs_n[:,agent,:],return_info=True) # batch x |A|
            pre_value = info['preactivation']
            if self.pre_activation_weight > 0.:
                pre_activation_policy_loss = (
                    (pre_value**2).sum(dim=1).mean()
                )
            else:
                pre_activation_policy_loss = torch.tensor(0.).to(ptu.device) 

            if self.use_automatic_entropy_tuning:
                alpha_loss = -(pis.detach() * self.log_alpha_n[agent].exp() * (torch.log(pis+1e-3) + self.target_entropy).detach()).sum(-1).mean()
                self.alpha_optimizer_n[agent].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer_n[agent].step()
                alpha = self.log_alpha_n[agent].exp()
            else:
                alpha_loss = torch.zeros(1)
                alpha = torch.ones(1)

            q_output = torch.zeros_like(pis).to(ptu.device)
            for a_i in range(self.env.action_space.n):
                current_actions = online_actions_n.detach().clone()
                action_i = torch.zeros(self.env.action_space.n).to(ptu.device)
                action_i[a_i] = 1.
                current_actions[:,agent,:] = action_i
                next_actions = torch.zeros_like(current_actions)
                for k in range(self.logit_level):
                    for agent_j in range(num_agent):
                        other_action_index = self.other_action_indices[agent_j]
                        current_other_actions = current_actions[:,other_action_index,:]
                        if agent_j != agent:
                            q_j = self.qf_n[agent_j](whole_obs,current_other_actions.view(batch_size, -1)).detach()
                            if self.double_q:
                                q2_j = self.qf2_n[agent_j](whole_obs,current_other_actions.view(batch_size, -1)).detach()
                                q_j = torch.min(q_j,q2_j)
                            action_j = F.gumbel_softmax(q_j, tau=alpha_n[agent_j], hard=self.gumbel_hard)
                            next_actions[:,agent_j,:] = action_j
                        else:
                            next_actions[:,agent_j,:] = action_i
                    current_actions = next_actions

                other_action_index = self.other_action_indices[agent]
                current_other_actions = current_actions[:,other_action_index,:]
                q_output_i = self.qf_n[agent](whole_obs, current_other_actions.view(batch_size, -1)).detach()
                q_output_i = torch.sum(q_output_i*action_i,dim=-1)
                if self.double_q:
                    q2_output_i = self.qf2_n[agent](whole_obs, current_other_actions.view(batch_size, -1)).detach()
                    q2_output_i = torch.sum(q2_output_i*action_i,dim=-1)
                    q_output_i = torch.min(q_output_i,q2_output_i)
                q_output[:,a_i] = q_output_i


            raw_policy_loss = (- pis * q_output).sum(-1).mean()
            entropy_loss = (pis * alpha * torch.log(pis+1e-3)).sum(-1).mean()
            policy_loss = (
                    raw_policy_loss +
                    pre_activation_policy_loss * self.pre_activation_weight +
                    entropy_loss
            )

            """
            Critic operations.
            """
            # speed up computation by not backpropping these gradients
            new_pis = self.policy_n[agent](next_obs_n[:,agent,:]).detach()
            next_current_actions = next_actions_n.clone()
            other_action_index = self.other_action_indices[agent]
            next_other_actions = next_current_actions[:,other_action_index,:].detach()
            other_actions = actions_n[:,other_action_index,:].detach()

            next_target_q_values = self.target_qf_n[agent](
                whole_next_obs,
                next_other_actions.view(batch_size,-1),
            )
            if self.double_q:
                next_target_q2_values = self.target_qf2_n[agent](
                    whole_next_obs,
                    next_other_actions.view(batch_size,-1),
                )
                next_target_q_values = torch.min(next_target_q_values, next_target_q2_values)

            next_target_q_values =  (new_pis*(next_target_q_values - alpha * torch.log(new_pis+1e-3))).sum(-1,keepdim=True) # batch
            q_target = self.reward_scale*rewards_n[:,agent,:] + (1. - terminals_n[:,agent,:]) * self.discount * next_target_q_values
            q_target = q_target.detach()
            q_target = torch.clamp(q_target, self.min_q_value, self.max_q_value)

            q_pred = torch.sum(self.qf_n[agent](whole_obs, other_actions.view(batch_size,-1))*actions_n[:,agent,:].detach(),dim=-1,keepdim=True)
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
                q2_pred = torch.sum(self.qf2_n[agent](whole_obs, other_actions.view(batch_size,-1))*actions_n[:,agent,:].detach(),dim=-1,keepdim=True)
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
            if self.clip_gradient > 0.:
                nn.utils.clip_grad_norm_(self.policy_n[agent].parameters(), self.clip_gradient)
            self.policy_optimizer_n[agent].step()

            self.qf_optimizer_n[agent].zero_grad()
            qf_loss.backward()
            if self.clip_gradient > 0.:
                nn.utils.clip_grad_norm_(self.qf_n[agent].parameters(), self.clip_gradient)
            self.qf_optimizer_n[agent].step()

            if self.double_q:
                self.qf2_optimizer_n[agent].zero_grad()
                qf2_loss.backward()
                if self.clip_gradient > 0.:
                    nn.utils.clip_grad_norm_(self.qf2_n[agent].parameters(), self.clip_gradient)
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
                self.eval_statistics['Preactivation Policy Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    pre_activation_policy_loss
                ))
                self.eval_statistics['Entropy Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    entropy_loss
                ))
                self.eval_statistics['Alpha {}'.format(agent)] = np.mean(ptu.get_numpy(
                    alpha
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Predictions {}'.format(agent),
                    ptu.get_numpy(q_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Targets {}'.format(agent),
                    ptu.get_numpy(q_target),
                ))
                if self.double_q:
                    self.eval_statistics.update(create_stats_ordered_dict(
                        'Q2 Predictions {}'.format(agent),
                        ptu.get_numpy(q2_pred),
                    ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Pis {}'.format(agent),
                    ptu.get_numpy(pis),
                ))
                
        self._need_to_update_eval_statistics = False
        self._update_target_networks()
        self._n_train_steps_total += 1

    def _update_target_networks(self):
        for qf, target_qf in \
            zip(self.qf_n, self.target_qf_n):
            if self.use_soft_update:
                ptu.soft_update_from_to(qf, target_qf, self.tau)
            else:
                if self._n_train_steps_total % self.target_hard_update_period == 0:
                    ptu.copy_model_params_from_to(qf, target_qf)
        if self.double_q:
            for qf2, target_qf2 in \
                zip(self.qf2_n, self.target_qf2_n):
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
        )
        if self.double_q:
            res['qf2_n'] = self.qf2_n
            res['target_qf2_n'] = self.target_qf2_n
        return res