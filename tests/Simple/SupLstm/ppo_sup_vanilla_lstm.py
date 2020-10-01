import gym
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.vpg.ppo import PPOTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm

def experiment(variant):
    from simple_sup_lstm import SimpleSupLSTMEnv
    expl_env = SimpleSupLSTMEnv(**variant['env_kwargs'])
    eval_env = SimpleSupLSTMEnv(**variant['env_kwargs'])
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    hidden_dim = variant['lstm_kwargs']['hidden_dim']
    max_path_length = variant['trainer_kwargs']['max_path_length']
    num_layers = variant['lstm_kwargs']['num_layers']
    a_0 = np.zeros(action_dim)
    h_0 = np.zeros(hidden_dim*num_layers)
    c_0 = np.zeros(hidden_dim*num_layers)
    decoder = torch.nn.Linear(hidden_dim, action_dim)
    from layers import ReshapeLayer
    sup_learner = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            ReshapeLayer(shape=(1, action_dim)),
        )
    from sup_softmax_lstm_policy import SupSoftmaxLSTMPolicy
    policy = SupSoftmaxLSTMPolicy(
                a_0=a_0,
                h_0=h_0,
                c_0=c_0,
                obs_dim=obs_dim,
                action_dim=action_dim,
                decoder=decoder,
                sup_learner=sup_learner,
                **variant['lstm_kwargs']
                )
    print('parameters: ',np.sum([p.view(-1).shape[0] for p in policy.parameters()]))

    vf = Mlp(
        hidden_sizes=[16, 16],
        input_size=obs_dim,
        output_size=1,
    )
    
    vf_criterion = nn.MSELoss()
    from rlkit.torch.policies.make_deterministic import MakeDeterministic
    eval_policy = MakeDeterministic(policy)
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
        observation_dim=obs_dim,
        action_dim=action_dim,
        label_dim=1,
        max_replay_buffer_size=int(1e6),
        max_path_length=max_path_length,
        recurrent=True,
    )

    from rlkit.torch.vpg.ppo_sup_vanilla import PPOSupVanillaTrainer
    trainer = PPOSupVanillaTrainer(
        policy=policy,
        value_function=vf,
        vf_criterion=vf_criterion,
        replay_buffer=replay_buffer,
        recurrent=True,
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
    parser.add_argument('--exp_name', type=str, default='SimpleSupLSTM')
    parser.add_argument('--obs', type=int, default=1)
    parser.add_argument('--int', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='PPOSupVanillaLSTM')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--sw', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="gap_and_last")
    parser.add_argument('--snapshot_gap', type=int, default=500)
    args = parser.parse_args()
    import os.path as osp
    pre_dir = './Data/'+args.exp_name+'obs'+str(args.obs)+'int'+str(args.int)
    main_dir = args.log_dir\
                +('layer'+str(args.layer))\
                +('hidden'+str(args.hidden))\
                +(('sw'+str(args.sw)) if args.sw else '')\
                +(('ep'+str(args.epoch)) if args.epoch else '')\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('bs'+str(args.bs)) if args.bs else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    max_path_length = 10
    # noinspection PyTypeChecker
    variant = dict(
        lstm_kwargs=dict(
            hidden_dim=args.hidden,
            num_layers=args.layer,
        ),
        env_kwargs=dict(
            obs_dim=args.obs,
            num_interval=args.int,
        ),
        algorithm_kwargs=dict(
            num_epochs=(args.epoch if args.epoch else 2000),
            num_eval_steps_per_epoch=1000,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=(args.bs if args.bs else 1000),
            max_path_length=max_path_length,
            save_best=True,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            max_path_length=max_path_length,
            policy_lr=(args.lr if args.lr else 1e-4),
            sup_lr=(args.lr if args.lr else 1e-4),
            vf_lr=(args.lr if args.lr else 1e-3),
            exploration_bonus=0.,
            # sup_weight=(args.sw if args.sw else 0.1),
            sup_batch_size=(args.bs if args.bs else 100),
        ),
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
