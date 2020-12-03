from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class PRGTrainer(TorchTrainer):
    """
    Probalistic Recursive Graph
    """
    def __init__(
            self,
            env,
            qf1_n,
            target_qf1_n,
            policy_n,
            target_policy_n,
            cactor_n,
            online_action,
            target_action,
            online_next_action,
            qf2_n,
            target_qf2_n,
            deterministic_cactor_in_graph=False,
            deterministic_next_action=False,
            prg_next_action=False,
            use_entropy_loss=False,
            use_entropy_reward=False,
            use_cactor_entropy_loss=False,
            use_automatic_entropy_tuning=True,
            state_dependent_alpha=False,
            log_alpha_n = None,
            log_calpha_n = None,
            target_entropy=None,

            logit_level=1,

            discount=0.99,
            reward_scale=1.0,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0.,
            init_alpha=1.,
            cactor_learning_rate=1e-4,
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
        self.qf1_n = qf1_n
        self.target_qf1_n = target_qf1_n
        self.qf2_n = qf2_n
        self.target_qf2_n = target_qf2_n
        self.policy_n = policy_n
        self.target_policy_n = target_policy_n
        self.cactor_n = cactor_n

        self.online_action = online_action
        self.target_action = target_action
        self.online_next_action = online_next_action
        self.logit_level = logit_level
        self.deterministic_cactor_in_graph = deterministic_cactor_in_graph
        self.deterministic_next_action = deterministic_next_action
        self.prg_next_action = prg_next_action

        self.discount = discount
        self.reward_scale = reward_scale

        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.cactor_learning_rate = cactor_learning_rate
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.qf_criterion = qf_criterion
        self.pre_activation_weight = pre_activation_weight
        self.min_q_value = min_q_value
        self.max_q_value = max_q_value

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

        self.cactor_optimizer_n = [
            optimizer_class(
                self.cactor_n[i].parameters(),
                lr=self.cactor_learning_rate,
            ) for i in range(len(self.cactor_n))]

        self.init_alpha = init_alpha
        self.use_entropy_loss = use_entropy_loss
        self.use_entropy_reward = use_entropy_reward
        self.use_cactor_entropy_loss = use_cactor_entropy_loss
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.state_dependent_alpha = state_dependent_alpha
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            if self.use_entropy_loss:
                # self.log_alpha_n = [ptu.zeros(1, requires_grad=True) for i in range(len(self.policy_n))]
                if self.state_dependent_alpha:
                    self.log_alpha_n = log_alpha_n
                    self.alpha_optimizer_n = [
                            optimizer_class(
                                self.log_alpha_n[i].parameters(),
                                lr=self.policy_learning_rate,
                            ) for i in range(len(self.log_alpha_n))]
                else:
                    self.log_alpha_n = [ptu.tensor([np.log(self.init_alpha)], requires_grad=True, dtype=torch.float32) for i in range(len(self.policy_n))]
                    self.alpha_optimizer_n = [
                        optimizer_class(
                            [self.log_alpha_n[i]],
                            lr=self.policy_learning_rate,
                        ) for i in range(len(self.log_alpha_n))]

            if self.use_cactor_entropy_loss:
                # self.log_calpha_n = [ptu.zeros(1, requires_grad=True) for i in range(len(self.policy_n))]
                if self.state_dependent_alpha:
                    self.log_calpha_n = log_calpha_n
                    self.calpha_optimizer_n = [
                            optimizer_class(
                                self.log_calpha_n[i].parameters(),
                                lr=self.policy_learning_rate,
                            ) for i in range(len(self.log_calpha_n))]
                else:
                    self.log_calpha_n = [ptu.tensor([np.log(self.init_alpha)], requires_grad=True, dtype=torch.float32) for i in range(len(self.policy_n))]
                    self.calpha_optimizer_n = [
                        optimizer_class(
                            [self.log_calpha_n[i]],
                            lr=self.policy_learning_rate,
                        ) for i in range(len(self.log_calpha_n))]

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

        """
        Policy operations.
        """
        online_actions_n, online_pre_values_n, online_log_pis_n = [], [], []
        for agent in range(num_agent):
            policy_actions, info = self.policy_n[agent](
                obs_n[:,agent,:], return_info=True,
            )
            online_actions_n.append(policy_actions)
            online_pre_values_n.append(info['preactivation'])
            online_log_pis_n.append(info['log_prob'])
        k0_actions = torch.stack(online_actions_n) # num_agent x batch x a_dim
        k0_actions = k0_actions.transpose(0,1).contiguous() # batch x num_agent x a_dim
        k1_actions = torch.zeros_like(k0_actions)
        for agent_j in range(num_agent):
            other_action_index = np.array([i for i in range(num_agent) if i!=agent_j])
            other_actions = k0_actions[:,other_action_index,:].view(batch_size,-1)
            if self.deterministic_cactor_in_graph:
                cactor_actions = self.cactor_n[agent_j](torch.cat((whole_obs,other_actions),dim=-1),deterministic=True)
            else:
                cactor_actions = self.cactor_n[agent_j](torch.cat((whole_obs,other_actions),dim=-1))
            k1_actions[:,agent_j,:] = cactor_actions

        policy_gradients_n = []
        for agent in range(num_agent):
            policy_actions = online_actions_n[agent]
            pre_value = online_pre_values_n[agent]
            log_pi = online_log_pis_n[agent]
            if self.pre_activation_weight > 0.:
                pre_activation_policy_loss = (
                    (pre_value**2).sum(dim=1).mean()
                )
            else:
                pre_activation_policy_loss = torch.tensor(0.).to(ptu.device) 
            if self.use_entropy_loss:
                if self.use_automatic_entropy_tuning:
                    if self.state_dependent_alpha:
                        alpha = self.log_alpha_n[agent](whole_obs).exp()
                    else:
                        alpha = self.log_alpha_n[agent].exp()
                    alpha_loss = -(alpha * (log_pi + self.target_entropy).detach()).mean()
                    self.alpha_optimizer_n[agent].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer_n[agent].step()
                    if self.state_dependent_alpha:
                        alpha = self.log_alpha_n[agent](whole_obs).exp().detach()
                    else:
                        alpha = self.log_alpha_n[agent].exp().detach()
                else:
                    alpha_loss = torch.tensor(0.).to(ptu.device)
                    alpha = torch.tensor(self.init_alpha).to(ptu.device)
                entropy_loss = (alpha*log_pi).mean()
            else:
                entropy_loss = torch.tensor(0.).to(ptu.device)

            input_actions = k1_actions.clone()
            input_actions[:,agent,:] = policy_actions
            q1_output = self.qf1_n[agent](whole_obs, input_actions.view(batch_size, -1))
            q2_output = self.qf2_n[agent](whole_obs, input_actions.view(batch_size, -1))
            q_output = torch.min(q1_output,q2_output)
            raw_policy_loss = -q_output.mean()
            policy_loss = (
                    raw_policy_loss +
                    pre_activation_policy_loss * self.pre_activation_weight +
                    entropy_loss
            )

            self.policy_optimizer_n[agent].zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_gradients_n.append([p.grad.clone() for p in self.policy_n[agent].parameters()])
            # self.policy_optimizer_n[agent].step()

            if self._need_to_update_eval_statistics:
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
                if self.use_entropy_loss:
                    if self.state_dependent_alpha:
                        self.eval_statistics.update(create_stats_ordered_dict(
                            'Alpha {}'.format(agent),
                            ptu.get_numpy(alpha),
                        ))
                    else:
                        self.eval_statistics['Alpha {} Mean'.format(agent)] = np.mean(ptu.get_numpy(
                            alpha
                        ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy Action {}'.format(agent),
                    ptu.get_numpy(policy_actions),
                ))

        for agent in range(num_agent):
            self.policy_optimizer_n[agent].zero_grad()
            for pid,p in enumerate(self.policy_n[agent].parameters()):
                p.grad = policy_gradients_n[agent][pid]
            self.policy_optimizer_n[agent].step()

        """
        Critic operations.
        """
        with torch.no_grad():
            next_actions_n, next_log_pis_n = [], []
            for agent in range(num_agent):
                next_actions, next_info = self.policy_n[agent](
                    next_obs_n[:,agent,:], return_info=True,
                    deterministic=self.deterministic_next_action,
                )
                next_actions_n.append(next_actions)
                next_log_pis_n.append(next_info['log_prob'])
            next_actions_n = torch.stack(next_actions_n) # num_agent x batch x a_dim
            next_actions_n = next_actions_n.transpose(0,1).contiguous() # batch x num_agent x a_dim

            if self.prg_next_action:
                k0_actions = next_actions_n
                k1_actions = torch.zeros_like(k0_actions)
                for agent_j in range(num_agent):
                    other_action_index = np.array([i for i in range(num_agent) if i!=agent_j])
                    other_actions = k0_actions[:,other_action_index,:].view(batch_size,-1)
                    if self.deterministic_cactor_in_graph:
                        cactor_actions = self.cactor_n[agent_j](torch.cat((whole_next_obs,other_actions),dim=-1),deterministic=True)
                    else:
                        cactor_actions = self.cactor_n[agent_j](torch.cat((whole_next_obs,other_actions),dim=-1))
                    k1_actions[:,agent_j,:] = cactor_actions

        for agent in range(num_agent):
            with torch.no_grad():
                if self.prg_next_action:
                    input_actions = k1_actions.clone()
                    input_actions[:,agent,:] = next_actions_n[:,agent,:]
                else:
                    input_actions = next_actions_n
                next_target_q1_values = self.target_qf1_n[agent](
                    whole_next_obs,
                    input_actions.view(batch_size,-1),
                )
                next_target_q2_values = self.target_qf2_n[agent](
                    whole_next_obs,
                    input_actions.view(batch_size,-1),
                )
                next_target_q_values = torch.min(next_target_q1_values, next_target_q2_values)

                if self.use_entropy_reward:
                    if self.state_dependent_alpha:
                        next_alpha = self.log_alpha_n[agent](whole_next_obs).exp()
                    else:
                        next_alpha = alpha
                    next_target_q_values =  next_target_q_values - next_alpha * next_log_pis_n[agent]

                q_target = self.reward_scale*rewards_n[:,agent,:] + (1. - terminals_n[:,agent,:]) * self.discount * next_target_q_values
                q_target = torch.clamp(q_target, self.min_q_value, self.max_q_value)

            q1_pred = self.qf1_n[agent](whole_obs, whole_actions)
            raw_qf1_loss = self.qf_criterion(q1_pred, q_target)
            if self.qf_weight_decay > 0:
                reg_loss1 = self.qf_weight_decay * sum(
                    torch.sum(param ** 2)
                    for param in self.qf1_n[agent].regularizable_parameters()
                )
                qf1_loss = raw_qf1_loss + reg_loss1
            else:
                qf1_loss = raw_qf1_loss

            q2_pred = self.qf2_n[agent](whole_obs, whole_actions)
            raw_qf2_loss = self.qf_criterion(q2_pred, q_target)
            if self.qf_weight_decay > 0:
                reg_loss2 = self.qf_weight_decay * sum(
                    torch.sum(param ** 2)
                    for param in self.qf2_n[agent].regularizable_parameters()
                )
                qf2_loss = raw_qf2_loss + reg_loss2
            else:
                qf2_loss = raw_qf2_loss

            self.qf1_optimizer_n[agent].zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer_n[agent].step()

            self.qf2_optimizer_n[agent].zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer_n[agent].step()

            if self._need_to_update_eval_statistics:
                self.eval_statistics['QF1 Loss {}'.format(agent)] = np.mean(ptu.get_numpy(qf1_loss))
                self.eval_statistics['QF2 Loss {}'.format(agent)] = np.mean(ptu.get_numpy(qf2_loss))
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

        """
        Central actor operations.
        """
        for agent in range(num_agent):
            other_action_index = np.array([i for i in range(num_agent) if i!=agent])
            other_actions = actions_n[:,other_action_index,:].view(batch_size,-1)
            cactor_actions, cactor_info = self.cactor_n[agent](
                torch.cat((whole_obs,other_actions),dim=-1), return_info=True,
            )
            cactor_pre_value = cactor_info['preactivation']
            if self.pre_activation_weight > 0:
                pre_activation_cactor_loss = (
                    (cactor_pre_value**2).sum(dim=1).mean()
                )
            else:
                pre_activation_cactor_loss = torch.tensor(0.).to(ptu.device)
            if self.use_cactor_entropy_loss:
                cactor_log_pi = cactor_info['log_prob']
                if self.use_automatic_entropy_tuning:
                    if self.state_dependent_alpha:
                        calpha = self.log_calpha_n[agent](whole_obs).exp()
                    else:
                        calpha = self.log_calpha_n[agent].exp()
                    calpha_loss = -(calpha * (cactor_log_pi + self.target_entropy).detach()).mean()
                    self.calpha_optimizer_n[agent].zero_grad()
                    calpha_loss.backward()
                    self.calpha_optimizer_n[agent].step()
                    if self.state_dependent_alpha:
                        calpha = self.log_calpha_n[agent](whole_obs).exp().detach()
                    else:
                        calpha = self.log_calpha_n[agent].exp().detach()
                else:
                    calpha_loss = torch.tensor(0.).to(ptu.device)
                    calpha = torch.tensor(self.init_alpha).to(ptu.device)
                cactor_entropy_loss = (calpha*cactor_log_pi).mean()
            else:
                cactor_entropy_loss = torch.tensor(0.).to(ptu.device)
            current_actions = actions_n.clone()
            current_actions[:,agent,:] = cactor_actions 
            q1_output = self.qf1_n[agent](whole_obs, current_actions.view(batch_size, -1))
            q2_output = self.qf2_n[agent](whole_obs, current_actions.view(batch_size, -1))
            q_output = torch.min(q1_output,q2_output)
            raw_cactor_loss = - q_output.mean()
            cactor_loss = (
                    raw_cactor_loss +
                    pre_activation_cactor_loss * self.pre_activation_weight +
                    cactor_entropy_loss
            )

            self.cactor_optimizer_n[agent].zero_grad()
            cactor_loss.backward()
            self.cactor_optimizer_n[agent].step()

            if self._need_to_update_eval_statistics:
                if self.use_cactor_entropy_loss:
                    if self.state_dependent_alpha:
                        self.eval_statistics.update(create_stats_ordered_dict(
                            'CAlpha {}'.format(agent),
                            ptu.get_numpy(calpha),
                        ))
                    else:
                        self.eval_statistics['CAlpha {} Mean'.format(agent)] = np.mean(ptu.get_numpy(
                            calpha
                        ))
                self.eval_statistics['Cactor Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    cactor_loss
                ))
                self.eval_statistics['Raw Cactor Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    raw_cactor_loss
                ))
                self.eval_statistics['Preactivation Cactor Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    pre_activation_cactor_loss
                ))
                self.eval_statistics['Entropy Cactor Loss {}'.format(agent)] = np.mean(ptu.get_numpy(
                    cactor_entropy_loss
                ))
                
        self._need_to_update_eval_statistics = False
        self._update_target_networks()
        self._n_train_steps_total += 1

    def _update_target_networks(self):
        for policy, target_policy, qf1, target_qf1, qf2, target_qf2 in \
            zip(self.policy_n, self.target_policy_n, self.qf1_n, self.target_qf1_n, self.qf2_n, self.target_qf2_n):
            if self.use_soft_update:
                ptu.soft_update_from_to(policy, target_policy, self.tau)
                ptu.soft_update_from_to(qf1, target_qf1, self.tau)
                ptu.soft_update_from_to(qf2, target_qf2, self.tau)
            else:
                if self._n_train_steps_total % self.target_hard_update_period == 0:
                    ptu.copy_model_params_from_to(policy, target_policy)
                    ptu.copy_model_params_from_to(qf1, target_qf1)
                    ptu.copy_model_params_from_to(qf2, target_qf2)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        res = [
            *self.policy_n,
            *self.target_policy_n,
            *self.cactor_n,
            *self.qf1_n,
            *self.target_qf1_n,
            *self.qf2_n,
            *self.target_qf2_n,
        ]
        if self.state_dependent_alpha:
            res.extend(self.log_alpha_n)
            if self.use_cactor_entropy_loss:
                res.extend(self.log_calpha_n)
        return res

    def get_snapshot(self):
        res = dict(
            qf1_n=self.qf1_n,
            target_qf1_n=self.target_qf1_n,
            qf2_n=self.qf2_n,
            target_qf2_n=self.target_qf2_n,
            cactor_n=self.cactor_n,
            trained_policy_n=self.policy_n,
            target_policy_n=self.target_policy_n,
        )
        if self.state_dependent_alpha:
            res['log_alpha_n']=self.log_alpha_n
            if self.use_cactor_entropy_loss:
                res['log_calpha_n']=self.log_calpha_n
        return res
