import gym
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm

def experiment(variant):
    from simple_sup import SimpleSupEnv
    expl_env = SimpleSupEnv(**variant['env_kwars'])
    eval_env = SimpleSupEnv(**variant['env_kwars'])
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    hidden_dim = variant['hidden_dim']
    encoder = nn.Sequential(
             nn.Linear(obs_dim,hidden_dim),
             nn.ReLU(),
             nn.Linear(hidden_dim,hidden_dim),
             nn.ReLU(),
            )
    decoder = nn.Linear(hidden_dim, action_dim)
    from layers import ReshapeLayer
    sup_learner = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            ReshapeLayer(shape=(1, action_dim)),
        )
    from sup_softmax_policy import SupSoftmaxPolicy
    policy = SupSoftmaxPolicy(encoder, decoder, sup_learner)
    print('parameters: ',np.sum([p.view(-1).shape[0] for p in policy.parameters()]))

    vf = Mlp(
        hidden_sizes=[32],
        input_size=obs_dim,
        output_size=1,
    )
    vf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(policy,use_preactivation=True)
    expl_policy = policy

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    from sup_replay_buffer import SupReplayBuffer
    replay_buffer = SupReplayBuffer(
        observation_dim = obs_dim,
        label_dim = 1,
        max_replay_buffer_size = int(1e6),
    )

    from rlkit.torch.vpg.ppo_sup import PPOSupTrainer
    trainer = PPOSupTrainer(
        policy=policy,
        value_function=vf,
        vf_criterion=vf_criterion,
        replay_buffer=replay_buffer,
        **variant['trainer_kwargs']
    )
    algorithm = TorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='SimpleSup')
    parser.add_argument('--obs', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='PPOSup')
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--sw', type=float, default=None)
    parser.add_argument('--eb', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="gap_and_last")
    parser.add_argument('--snapshot_gap', type=int, default=500)
    args = parser.parse_args()
    import os.path as osp
    pre_dir = './Data/'+args.exp_name+'obs'+str(args.obs)
    main_dir = args.log_dir\
                +('hidden'+str(args.hidden))\
                +(('sw'+str(args.sw)) if args.sw else '')\
                +(('eb'+str(args.eb)) if args.eb else '')\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('bs'+str(args.bs)) if args.bs else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    max_path_length = 2
    # noinspection PyTypeChecker
    variant = dict(
        env_kwars=dict(
            num_obs=args.obs
        ),
        algorithm_kwargs=dict(
            num_epochs=(args.epoch if args.epoch else 1000),
            num_eval_steps_per_epoch=1000,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=(args.bs if args.bs else 10),
            max_path_length=max_path_length,
            save_best=False,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            max_path_length=max_path_length,
            policy_lr=(args.lr if args.lr else 1e-4),
            vf_lr=(args.lr if args.lr else 1e-3),
            exploration_bonus=(args.eb if args.eb else 0.),
            sup_weight=(args.sw if args.sw else 0.1),
            sup_batch_size=(args.bs if args.bs else 10),
        ),
        hidden_dim = args.hidden,
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
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
