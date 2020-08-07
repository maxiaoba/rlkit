import copy
import gym
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.my_dqn import DQNTrainer
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.prioritized_replay_buffer import PrioritizedReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

def experiment(variant):
    from cartpole import CartPoleEnv
    expl_env = CartPoleEnv(mode=2)
    eval_env = CartPoleEnv(mode=2)
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    qf = Mlp(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['qf_kwargs']
    )
    target_qf = copy.deepcopy(qf)
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space, variant['epsilon']),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    replay_buffer = PrioritizedReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    qf_criterion = nn.MSELoss()
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        replay_buffer=replay_buffer,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='Cartpole')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--log_dir', type=str, default='MyDQN')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--cg', type=float, default=None)
    parser.add_argument('--rs', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--expl', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="gap_and_last")
    parser.add_argument('--snapshot_gap', type=int, default=500)
    args = parser.parse_args()
    import os.path as osp
    pre_dir = './Data/'+args.exp_name
    main_dir = args.log_dir\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('cg'+str(args.cg)) if args.cg else '')\
                +(('rs'+str(args.rs)) if args.rs else '')\
                +(('bs'+str(args.bs)) if args.bs else '')\
                +(('expl'+str(args.expl)) if args.expl else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    # noinspection PyTypeChecker
    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=(args.epoch if args.epoch else 200),
            num_eval_steps_per_epoch=500,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=100,
            max_path_length=100,
            batch_size=(args.bs if args.bs else 256),
            save_best=True,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=(args.lr if args.lr else 1E-3),
            clip_gradient=(args.cg if args.cg else 0.),
            reward_scale=(args.rs if args.rs else 1.),
            soft_target_tau=1e-3,
            target_update_period=1,
        ),
        qf_kwargs=dict(
            hidden_sizes=[32,32],
        ),
        epsilon=(args.expl if args.expl else 0.1),
        replay_buffer_size=int(1E6),
    )
    import os
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir,'variant.json'),'w') as out_json:
        import json
        json.dump(variant,out_json,indent=2)
    import sys
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(osp.join(log_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    setup_logger(args.exp_name+'/'+main_dir, variant=variant,
                snapshot_mode=args.snapshot_mode, snapshot_gap=args.snapshot_gap,
                log_dir=log_dir)
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        ptu.set_gpu_mode(True)
    experiment(variant)
